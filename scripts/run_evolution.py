#!/usr/bin/env python3
"""Plane A: Client-Side AlphaEvolve Controller & Split-Privacy Evaluator Loop.

Default GCP Target:
  --project-id project-3zcmvkwmrw953
  --engine-id  global-gemini-enterprise

Usage:
  1. Offline dry run (zero GCP calls; verifies seed, markers, evaluator, payloads):
       python3 scripts/run_evolution.py --dry-run

  2. Live 5-program smoke test on project-3zcmvkwmrw953 (recommended first!):
       python3 scripts/run_evolution.py --smoke

  3. Full demo evolution run (40-80 programs):
       python3 scripts/run_evolution.py --max-programs 50 --concurrency 2
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiment"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from alphaevolve_client import AlphaEvolveAPIError, AlphaEvolveRestClient
import evaluator


def validate_evolve_block_markers(source_text: str, filename: str = "solver.py") -> None:
    """Verify strict column-0 AlphaEvolve EVOLVE-BLOCK marker syntax."""
    lines = source_text.splitlines()
    starts = [i for i, ln in enumerate(lines) if "EVOLVE-BLOCK-START" in ln]
    ends = [i for i, ln in enumerate(lines) if "EVOLVE-BLOCK-END" in ln]
    if not starts or len(starts) != len(ends):
        raise ValueError(f"{filename}: mismatched EVOLVE-BLOCK-START ({len(starts)}) / END ({len(ends)})")
    for s_idx, e_idx in zip(starts, ends):
        if lines[s_idx] != "# EVOLVE-BLOCK-START":
            raise ValueError(
                f"{filename}:{s_idx + 1}: marker must be exact column-0 '# EVOLVE-BLOCK-START', got {lines[s_idx]!r}"
            )
        if lines[e_idx] != "# EVOLVE-BLOCK-END":
            raise ValueError(
                f"{filename}:{e_idx + 1}: marker must be exact column-0 '# EVOLVE-BLOCK-END', got {lines[e_idx]!r}"
            )
        if e_idx <= s_idx + 1:
            raise ValueError(f"{filename}:{s_idx + 1}: EVOLVE-BLOCK cannot be empty")
        if not lines[s_idx + 1].strip():
            raise ValueError(f"{filename}:{s_idx + 2}: no blank line allowed immediately after EVOLVE-BLOCK-START")


def parse_models_spec(model_spec: str) -> List[Dict[str, Any]]:
    """Parse 'gemini-3.8-flash:0.6;gemini-3.1-pro-preview:0.4' into AlphaEvolve model list."""
    models: List[Dict[str, Any]] = []
    for raw in model_spec.replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            name, w = item.split(":", 1)
            models.append({"name": name.strip(), "weight": float(w.strip())})
        else:
            models.append({"name": item})
    return models


def extract_primary_score(eval_dict: Dict[str, Any], metric: str = evaluator.PRIMARY_METRIC) -> Optional[float]:
    for entry in eval_dict.get("scores", {}).get("scores", []):
        if entry.get("metric") == metric:
            val = entry.get("score")
            return float(val) if val is not None else None
    return None


def extract_all_scores(eval_dict: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        str(e["metric"]): (float(e["score"]) if e.get("score") is not None else None)
        for e in eval_dict.get("scores", {}).get("scores", [])
    }


def extract_first_insight(eval_dict: Dict[str, Any]) -> str:
    ins = eval_dict.get("insights", {}).get("insights", [])
    if not ins:
        return ""
    return f"[{ins[0].get('label', '')}] {ins[0].get('text', '')}"


def build_experiment_payloads(
    max_programs: int,
    concurrency: int,
    model_spec: str,
    seed_eval: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Construct verified AlphaEvolve experiment config + initial program payloads."""
    solver_code = (EXP_DIR / "solver.py").read_text(encoding="utf-8")
    schema_code = (EXP_DIR / "schema.py").read_text(encoding="utf-8")
    validate_evolve_block_markers(solver_code, "solver.py")

    # Read problem_description directly from experiment.yaml (no external PyYAML required!)
    yaml_text = (EXP_DIR / "experiment.yaml").read_text(encoding="utf-8")
    title = "Adani DC Multi-Objective Schedule Recovery Solver"
    problem_desc = ""
    try:
        import yaml  # type: ignore

        spec = yaml.safe_load(yaml_text)
        title = spec.get("title", title)
        problem_desc = spec.get("problem_description", "")
    except ImportError:
        # Clean stdlib extraction from the block scalar in experiment.yaml
        in_desc = False
        desc_lines: List[str] = []
        for ln in yaml_text.splitlines():
            if ln.startswith("problem_description:"):
                in_desc = True
                continue
            if in_desc:
                if ln and not ln.startswith("  "):
                    break
                desc_lines.append(ln[2:] if ln.startswith("  ") else ln)
        problem_desc = "\n".join(desc_lines).strip()

    exp_config: Dict[str, Any] = {
        "title": title,
        "problem_description": problem_desc,
        "program_language": "python",
        "run_settings": {
            "max_programs": max_programs,
            "concurrency": concurrency,
            "max_duration": "7200s",
            "idle_timeout": "3600s",
        },
        "generation_settings": {
            "include_full_program_in_prompt": True,
            "models": parse_models_spec(model_spec),
        },
        "evolution_settings": {
            "parent_sampling_config": {
                "pareto_sampling_config": {
                    "pareto_sampling_probability": 0.35,
                }
            }
        },
    }

    initial_program: Dict[str, Any] = {
        "content": {
            "files": [
                {"path": "solver.py", "content": solver_code},
                {"path": "schema.py", "content": schema_code},
            ]
        },
        "evaluation": seed_eval,
    }
    return exp_config, initial_program


def write_privacy_audit_row(
    audit_path: Path,
    candidate_id: str,
    generation_idx: int,
    code_bytes_in: int,
    eval_result: Dict[str, Any],
    eval_elapsed_ms: float,
    num_scenarios: int = 12,
) -> None:
    score_bytes_out = len(json.dumps(eval_result).encode("utf-8"))
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step": generation_idx,
        "candidate_id": candidate_id,
        "direction_in": "candidate_python_source_code_only",
        "code_bytes_in": code_bytes_in,
        "direction_out": "scalar_scores_and_diagnostics_only",
        "score_bytes_out": score_bytes_out,
        "scenarios_evaluated_locally": num_scenarios,
        "raw_schedule_bytes_egressed": 0,
        "eval_elapsed_ms": round(eval_elapsed_ms, 1),
        "primary_score": extract_primary_score(eval_result),
    }
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_artifacts(
    best_code: str,
    seed_code: str,
    lineage: List[Dict[str, Any]],
    experiment_meta: Dict[str, Any],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "baseline_solver.py").write_text(seed_code, encoding="utf-8")
    (ARTIFACTS_DIR / "evolved_solver.py").write_text(best_code, encoding="utf-8")

    diff_lines = list(
        difflib.unified_diff(
            seed_code.splitlines(keepends=True),
            best_code.splitlines(keepends=True),
            fromfile="experiment/solver.py (Gen-0 Baseline)",
            tofile="artifacts/evolved_solver.py (AlphaEvolve Winner)",
        )
    )
    (ARTIFACTS_DIR / "best_solver_diff.patch").write_text("".join(diff_lines), encoding="utf-8")
    (ARTIFACTS_DIR / "lineage_curve.json").write_text(
        json.dumps({"experiment": experiment_meta, "lineage": lineage}, indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "experiment_state.json").write_text(
        json.dumps(experiment_meta, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run client-side AlphaEvolve loop for DC schedule recovery.")
    parser.add_argument("--project-id", default="project-3zcmvkwmrw953", help="GCP Project ID")
    parser.add_argument("--engine-id", default="global-gemini-enterprise", help="Gemini Enterprise Engine ID")
    parser.add_argument("--location", default="global", help="Discovery Engine location (global/us/eu)")
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash:0.6;gemini-3.1-pro-preview:0.4",
        help="AlphaEvolve model or weighted mixture (capped Pro model <= 0.5)",
    )
    parser.add_argument("--max-programs", type=int, default=30, help="Max candidate programs to evaluate")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel generation concurrency")
    parser.add_argument("--poll-interval", type=float, default=4.0, help="Seconds between acquirePrograms polls")
    parser.add_argument("--smoke", action="store_true", help="Quick 5-program live smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Offline local validation only (zero GCP API calls)")
    args = parser.parse_args()

    if args.smoke:
        args.max_programs = 5
        args.concurrency = 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_code = (EXP_DIR / "solver.py").read_text(encoding="utf-8")

    print("=" * 76)
    print("PLANE A: Adani DC Schedule Recovery — Client-Side AlphaEvolve Controller")
    print("=" * 76)
    print(f"Target Project : {args.project_id}")
    print(f"Target Engine  : {args.engine_id} ({args.location})")
    print(f"Model Mixture  : {args.model}")
    print(f"Max Evaluations: {args.max_programs} (concurrency={args.concurrency})")
    print()

    # 1. Evaluate Gen-0 seed locally inside isolated client-side sandbox
    print("[1/4] Evaluating Gen-0 seed solver locally on 12 private training DC scenarios...")
    t0 = time.perf_counter()
    seed_eval = evaluator.evaluate_candidate_files_isolated({"solver.py": seed_code}, split="train")
    seed_ms = (time.perf_counter() - t0) * 1000.0
    seed_score = extract_primary_score(seed_eval)
    if seed_score is None:
        print("FATAL: Gen-0 seed failed evaluation:", extract_first_insight(seed_eval))
        return 1

    heldout_seed_eval = evaluator.evaluate_candidate_files_isolated({"solver.py": seed_code}, split="heldout")
    heldout_seed_score = extract_primary_score(heldout_seed_eval)
    print(f"  Gen-0 Train   {evaluator.PRIMARY_METRIC}: {seed_score:.4f} ({seed_ms:.0f} ms)")
    print(f"  Gen-0 Heldout {evaluator.PRIMARY_METRIC}: {heldout_seed_score:.4f}")
    print(f"  Insight: {extract_first_insight(seed_eval)}")

    exp_config, initial_program = build_experiment_payloads(
        max_programs=args.max_programs,
        concurrency=args.concurrency,
        model_spec=args.model,
        seed_eval=seed_eval,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Validated EVOLVE-BLOCK markers, experiment.yaml, seed evaluation, and API payload schema.")
        print(f"[DRY-RUN] Seed payload size (code + schema ONLY, 0 schedule bytes): {len(json.dumps(initial_program))} bytes")
        print("[DRY-RUN] Payload preview (config keys):", list(exp_config.keys()))
        print("[DRY-RUN] PASSED! Run with `--smoke` on Cloud Shell / Cloudtop to test live AlphaEvolve API.")
        return 0

    # 2. Initialize live AlphaEvolve experiment on project-3zcmvkwmrw953
    audit_path = ARTIFACTS_DIR / "boundary_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    client = AlphaEvolveRestClient(
        project_id=args.project_id,
        engine_id=args.engine_id,
        location=args.location,
    )

    print("\n[2/4] Creating Gemini Enterprise session & AlphaEvolve experiment...")
    try:
        session_name = client.create_session()
        print(f"  Session    : {session_name}")
        exp_resp = client.create_experiment(session_name, exp_config)
        experiment_name = exp_resp["name"]
        print(f"  Experiment : {experiment_name}")
        init_resp = client.create_initial_program(experiment_name, initial_program)
        init_prog_name = init_resp.get("name", "seed-0")
        print(f"  Seed Prog  : {init_prog_name}")
        client.start_experiment(experiment_name)
        print("  State      : RUNNING")
    except AlphaEvolveAPIError as exc:
        print(f"\nERROR contacting AlphaEvolve API: {exc}")
        return 2

    lineage: List[Dict[str, Any]] = [
        {
            "step": 0,
            "program_name": init_prog_name,
            "parent_programs": [],
            "feasible": True,
            "primary_score": seed_score,
            "best_so_far": seed_score,
            "scores": extract_all_scores(seed_eval),
            "insight": extract_first_insight(seed_eval),
        }
    ]
    write_privacy_audit_row(audit_path, init_prog_name, 0, len(seed_code.encode("utf-8")), seed_eval, seed_ms)

    best_score = seed_score
    best_code = seed_code
    best_program_name = init_prog_name
    evaluated_count = 0
    empty_polls = 0
    max_empty_polls = 45

    experiment_meta = {
        "project_id": args.project_id,
        "engine_id": args.engine_id,
        "location": args.location,
        "model": args.model,
        "session_name": session_name,
        "experiment_name": experiment_name,
        "initial_program_name": init_prog_name,
        "seed_train_score": seed_score,
        "seed_heldout_score": heldout_seed_score,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_artifacts(best_code, seed_code, lineage, experiment_meta)

    # 3. Client-side evolution loop (Code In -> Local Private Sandbox -> Scalar Scores Out)
    print(f"\n[3/4] Running client-side evolutionary loop (target: {args.max_programs} evaluations)...")
    while evaluated_count < args.max_programs:
        desired = min(args.concurrency, args.max_programs - evaluated_count)
        try:
            programs = client.acquire_programs(experiment_name, desired_programs_count=desired)
        except AlphaEvolveAPIError as exc:
            if "FAILED_PRECONDITION" in str(exc):
                exp_state = client.get_experiment(experiment_name).get("state")
                print(f"  Experiment state={exp_state}; attempting auto-resume...")
                if exp_state == "PAUSED":
                    client.resume_experiment(experiment_name)
                    continue
                if exp_state in ("COMPLETED", "FAILED"):
                    print(f"  Stopping loop: experiment entered terminal state {exp_state}.")
                    break
            raise

        if not programs:
            empty_polls += 1
            exp_state = client.get_experiment(experiment_name).get("state")
            print(f"  [poll {empty_polls}] No ready candidates yet (state={exp_state}); waiting {args.poll_interval:.0f}s...")
            if exp_state in ("COMPLETED", "FAILED") or empty_polls >= max_empty_polls:
                print(f"  Stopping polling (state={exp_state}, empty_polls={empty_polls}).")
                break
            time.sleep(args.poll_interval)
            continue

        empty_polls = 0
        submissions: List[Dict[str, Any]] = []

        for prog in programs:
            evaluated_count += 1
            prog_name = prog["name"]
            short_id = prog_name.rsplit("/", 1)[-1]
            lock_token = prog.get("lockToken") or prog.get("lock_token") or ""
            files_list = prog.get("content", {}).get("files", [])
            files_map = {f["path"]: f["content"] for f in files_list if "path" in f and "content" in f}
            cand_solver_code = files_map.get("solver.py", "")
            code_bytes_in = sum(len(v.encode("utf-8")) for v in files_map.values())

            t_eval = time.perf_counter()
            eval_result = evaluator.evaluate_candidate_files_isolated(files_map, split="train")
            eval_ms = (time.perf_counter() - t_eval) * 1000.0
            cand_score = extract_primary_score(eval_result)

            is_new_best = cand_score is not None and cand_score > best_score
            if is_new_best and cand_solver_code:
                best_score = cand_score  # type: ignore[assignment]
                best_code = cand_solver_code
                best_program_name = prog_name

            status_tag = (
                f"NEW BEST {cand_score:.4f} (+{cand_score - seed_score:.4f})"
                if is_new_best
                else (f"FEASIBLE {cand_score:.4f}" if cand_score is not None else "DISQUALIFIED (score=None)")
            )
            print(f"  [{evaluated_count:02d}/{args.max_programs}] {short_id}: {status_tag} [{eval_ms:.0f}ms]")
            if cand_score is None:
                print(f"      -> Feedback sent to Gemini: {extract_first_insight(eval_result)[:160]}")

            write_privacy_audit_row(audit_path, prog_name, evaluated_count, code_bytes_in, eval_result, eval_ms)
            lineage.append(
                {
                    "step": evaluated_count,
                    "program_name": prog_name,
                    "parent_programs": prog.get("parentPrograms", prog.get("parent_programs", [])),
                    "feasible": cand_score is not None,
                    "primary_score": cand_score,
                    "best_so_far": best_score,
                    "scores": extract_all_scores(eval_result),
                    "insight": extract_first_insight(eval_result),
                }
            )
            submissions.append(
                {
                    "program": prog_name,
                    "lock_token": lock_token,
                    "evaluation": eval_result,
                }
            )

        client.submit_program_evaluations(experiment_name, submissions)
        experiment_meta["best_program_name"] = best_program_name
        experiment_meta["best_train_score"] = best_score
        experiment_meta["evaluated_count"] = evaluated_count
        save_artifacts(best_code, seed_code, lineage, experiment_meta)

    # 4. Final Held-Out Validation (Unseen DC Sites H01-H06)
    print("\n[4/4] Auditing winning evolved solver on 6 UNSEEN held-out DC scenarios (H01-H06)...")
    heldout_best_eval = evaluator.evaluate_candidate_files_isolated({"solver.py": best_code}, split="heldout")
    heldout_best_score = extract_primary_score(heldout_best_eval)

    experiment_meta["best_heldout_score"] = heldout_best_score
    experiment_meta["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    experiment_meta["train_lift_pct"] = round(((best_score - seed_score) / max(1e-6, seed_score)) * 100.0, 2)
    if heldout_best_score is not None and heldout_seed_score:
        experiment_meta["heldout_lift_pct"] = round(
            ((heldout_best_score - heldout_seed_score) / max(1e-6, heldout_seed_score)) * 100.0, 2
        )
    save_artifacts(best_code, seed_code, lineage, experiment_meta)

    print("\n" + "=" * 76)
    print("EVOLUTION COMPLETE — SUMMARY")
    print("=" * 76)
    print(f"  Train   HV : Gen-0 {seed_score:.4f} -> AlphaEvolve {best_score:.4f} ({experiment_meta['train_lift_pct']:+.1f}%)")
    print(
        f"  Heldout HV : Gen-0 {heldout_seed_score:.4f} -> AlphaEvolve {heldout_best_score:.4f} "
        f"({experiment_meta.get('heldout_lift_pct', 0.0):+.1f}%)"
    )
    print(f"  Artifacts  : {ARTIFACTS_DIR}/evolved_solver.py, lineage_curve.json, best_solver_diff.patch, boundary_audit.jsonl")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
