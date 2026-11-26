#!/usr/bin/env python3
"""Fair, apples-to-apples Pure LLM (Gemini) DC Recovery Baseline + Verifier Audit.

Sends the exact same held-out Data Centre scenario JSON (H01 / H02 / etc.) to
Vertex AI Gemini (default: gemini-3.8-flash on global) with strict structured JSON output and
audits Gemini's returned schedule through the EXACT same 5-gate `verifier.py`
used to grade AlphaEvolve solvers.

Works with zero extra pip packages (standard-library `urllib.request` + gcloud
access token against Vertex AI `generateContent` REST endpoint).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiment"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from alphaevolve_client import build_ssl_context, get_gcp_access_token
import evaluator
from schema import RecoveryPlan, Scenario, TaskAssignment
from verifier import compute_unmitigated_rfs_and_ld, verify_plan


def build_fair_llm_prompt(scenario: Scenario, prior_violations: Optional[List[str]] = None) -> str:
    """Build a fair, clear, complete scheduling prompt containing every constraint."""
    tasks = scenario.active_tasks()
    fim_arrivals = scenario.effective_fim_arrivals()
    rework_ids = {t.task_id for t in scenario.disruption.added_rework_tasks}

    compact_tasks = []
    for t in tasks:
        is_frozen = (t.baseline_end_day <= scenario.disruption.trigger_day) and (t.task_id not in rework_ids)
        compact_tasks.append(
            {
                "task_id": t.task_id,
                "zone": t.work_front_zone,
                "predecessors": list(t.predecessors),
                "fim_item_id": t.fim_item_id,
                "frozen": is_frozen,
                "baseline_start": t.baseline_start_day,
                "baseline_end": t.baseline_end_day,
                "modes": [
                    {
                        "mode": m.mode,
                        "dur": scenario.effective_duration(t, m),
                        "cost_cr": scenario.effective_cost(t, m),
                        "crew": m.crew_by_trade,
                    }
                    for m in t.available_modes
                ],
            }
        )

    payload = {
        "site": scenario.constraints.site_name,
        "disruption": scenario.disruption.summary,
        "trigger_day": scenario.disruption.trigger_day,
        "target_rfs_day": scenario.constraints.target_rfs_day,
        "ld_grace_days": scenario.constraints.ld_grace_days,
        "ld_rate_inr_cr_per_day": scenario.constraints.ld_rate_inr_cr_per_day,
        "max_recovery_budget_inr_cr": scenario.constraints.max_recovery_budget_inr_cr,
        "kronos_daily_cap": scenario.constraints.kronos_daily_cap,
        "manpower_cap_overrides_day_range": scenario.disruption.manpower_cap_overrides,
        "zone_max_concurrent": scenario.constraints.zone_max_concurrent,
        "effective_fim_arrival_day": fim_arrivals,
        "tasks": compact_tasks,
    }

    retry_block = ""
    if prior_violations:
        retry_block = (
            "\n\nYOUR PREVIOUS SCHEDULE FAILED THE DETERMINISTIC VERIFIER WITH THESE VIOLATIONS:\n"
            + "\n".join(f"  - {v}" for v in prior_violations[:12])
            + "\nFix EVERY violation above while keeping all other constraints satisfied."
        )

    return (
        "You are an expert hyperscale Data Centre capital-project scheduler.\n"
        "Given the disrupted Primavera P6 project JSON below, output ONE optimal, strictly feasible "
        "recovery schedule assigning EVERY task_id a `start_day`, `end_day`, and `selected_mode`.\n\n"
        "HARD CONSTRAINTS (ZERO VIOLATIONS ALLOWED):\n"
        "1. Include every single task_id from `tasks` exactly once.\n"
        "2. Any task with `frozen: true` MUST keep exact `baseline_start`, `baseline_end`, and mode `NORMAL`.\n"
        "3. Non-frozen tasks must satisfy `end_day - start_day == mode.dur` for the selected mode and "
        "`start_day >= trigger_day` (or `>= baseline_start` if already started before trigger_day).\n"
        "4. Finish-to-Start precedence: for every predecessor `p`, `start_day[task] >= end_day[p]`.\n"
        "5. FIM arrival gate: if `fim_item_id` is not null, `start_day >= effective_fim_arrival_day[fim_item_id]`.\n"
        "6. Kronos daily trade headcount caps: on EVERY day `d >= trigger_day` and for EVERY trade, the sum of "
        "`crew[trade]` across all tasks active on day `d` (`start_day <= d < end_day`) MUST be `<= kronos_daily_cap[trade]` "
        "(respecting any reduced cap inside `manpower_cap_overrides_day_range`).\n"
        "7. Work-front concurrency: on EVERY day `d >= trigger_day`, active tasks in each `zone` MUST be `<= zone_max_concurrent[zone]`.\n"
        "8. Budget: sum of `cost_cr` across all non-frozen tasks MUST be `<= max_recovery_budget_inr_cr`.\n\n"
        "OBJECTIVE: Minimize RFS completion day (`max(end_day)`) and total cost (recovery + LD).\n"
        "Return ONLY valid JSON matching the required response schema.\n"
        f"{retry_block}\n\n"
        f"SCENARIO JSON:\n{json.dumps(payload)}"
    )


def _parse_json_or_recover_assignments(text: str, finish_reason: str = "") -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse strict JSON schedule, or recover partial assignments if LLM truncated mid-output."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        # Recover any complete task assignment dicts emitted before truncation
        recovered: List[Dict[str, Any]] = []
        obj_pat = re.compile(r"\{[^{}]*\"task_id\"[^{}]*\}")
        for m in obj_pat.finditer(text):
            try:
                item = json.loads(m.group(0))
                if all(k in item for k in ("task_id", "start_day", "end_day", "selected_mode")):
                    recovered.append(item)
            except json.JSONDecodeError:
                continue
        reason = finish_reason or "JSON_TRUNCATED"
        note = (
            f"LLM response truncated ({reason}, {len(recovered)} complete task assignments emitted "
            f"before malformed JSON: {exc.msg})"
        )
        return {"plan_label": "PureLLM_Truncated", "assignments": recovered}, note


def call_vertex_gemini_json(
    project_id: str,
    location: str,
    model_name: str,
    prompt: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, Optional[str]]:
    """Invoke Vertex AI Gemini generateContent with strict JSON responseSchema."""
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "plan_label": {"type": "STRING"},
            "assignments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "task_id": {"type": "STRING"},
                        "start_day": {"type": "INTEGER"},
                        "end_day": {"type": "INTEGER"},
                        "selected_mode": {
                            "type": "STRING",
                            "enum": ["NORMAL", "OVERTIME", "DOUBLE_SHIFT", "ALT_SUBCONTRACTOR"],
                        },
                    },
                    "required": ["task_id", "start_day", "end_day", "selected_mode"],
                },
            },
        },
        "required": ["plan_label", "assignments"],
    }

    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    url = (
        f"https://{host}/v1/projects/{project_id}/locations/{location}"
        f"/publishers/google/models/{model_name}:generateContent"
    )
    gen_config: Dict[str, Any] = {
        "temperature": 0.1,
        "maxOutputTokens": 65536,
        "responseMimeType": "application/json",
        "responseSchema": response_schema,
    }
    # Gemini 3.x models default to HIGH thinking, which can consume output budget
    # and truncate 100-task JSON schedules; use LOW thinking for fast, complete JSON.
    if model_name.startswith("gemini-3"):
        gen_config["thinkingConfig"] = {"thinkingLevel": "LOW"}

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    headers = {
        "Authorization": f"Bearer {get_gcp_access_token()}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
    }

    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180.0, context=build_ssl_context()) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_txt = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vertex AI HTTP {exc.code} on {url}: {err_txt[:1000]}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    usage = raw.get("usageMetadata", {})
    candidates = raw.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {raw}")
    finish_reason = str(candidates[0].get("finishReason", ""))
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p and not p.get("thought")]
    if not text_parts:
        text_parts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    if not text_parts:
        raise RuntimeError(f"Gemini returned no text parts: {candidates[0]}")
    text = "".join(text_parts).strip()
    parsed, truncation_note = _parse_json_or_recover_assignments(text, finish_reason=finish_reason)
    return parsed, usage, elapsed_ms, truncation_note


def run_llm_on_scenario(
    scenario: Scenario,
    project_id: str,
    location: str,
    model_name: str,
    max_attempts: int = 1,
) -> Dict[str, Any]:
    """Run Pure Gemini directly on one held-out scenario and audit via verifier.py."""
    unmit_rfs, unmit_ld = compute_unmitigated_rfs_and_ld(scenario)
    attempts_log: List[Dict[str, Any]] = []
    prior_violations: Optional[List[str]] = None
    total_latency_ms = 0.0
    final_result: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        prompt = build_fair_llm_prompt(scenario, prior_violations=prior_violations)
        try:
            parsed, usage, elapsed_ms, truncation_note = call_vertex_gemini_json(
                project_id, location, model_name, prompt
            )
            total_latency_ms += elapsed_ms
            asgn_map: Dict[str, TaskAssignment] = {}
            for row in parsed.get("assignments", []):
                tid = str(row["task_id"])
                asgn_map[tid] = TaskAssignment(
                    task_id=tid,
                    start_day=int(row["start_day"]),
                    end_day=int(row["end_day"]),
                    selected_mode=row["selected_mode"],
                )
            plan = RecoveryPlan(
                plan_label=str(parsed.get("plan_label", f"PureLLM_{model_name}_try{attempt}")),
                assignments=asgn_map,
            )
            pm = verify_plan(scenario, plan, unmitigated_rfs=unmit_rfs, unmitigated_ld=unmit_ld)
            gate_counts: Dict[str, int] = {}
            for v in pm.violations:
                gate_counts[v.gate] = gate_counts.get(v.gate, 0) + 1
            sample_viols = [f"[{v.gate}] {v.message}" for v in pm.violations[:8]]
            if truncation_note:
                sample_viols = [f"[COMPLETENESS] {truncation_note}"] + sample_viols[:7]
                if "COMPLETENESS" not in gate_counts:
                    gate_counts["COMPLETENESS"] = 1

            attempt_summary = {
                "attempt": attempt,
                "feasible": pm.feasible and not truncation_note,
                "latency_ms": round(elapsed_ms, 1),
                "usage": usage,
                "rfs_day": pm.rfs_day,
                "rfs_days_recovered": pm.rfs_days_recovered,
                "incremental_cost_inr_cr": pm.incremental_cost_inr_cr,
                "ld_cost_inr_cr": pm.ld_cost_inr_cr,
                "ld_avoided_inr_cr": pm.ld_avoided_inr_cr,
                "mean_churn": pm.mean_churn,
                "violation_count": len(pm.violations) if pm.violations else (1 if truncation_note else 0),
                "violations_by_gate": gate_counts,
                "sample_violations": sample_viols,
            }
            attempts_log.append(attempt_summary)
            final_result = attempt_summary
            if pm.feasible:
                break
            prior_violations = attempt_summary["sample_violations"]
        except Exception as exc:
            err_summary = {
                "attempt": attempt,
                "feasible": False,
                "latency_ms": 0.0,
                "usage": {},
                "violation_count": 1,
                "violations_by_gate": {"LLM_PARSE_OR_API_ERROR": 1},
                "sample_violations": [str(exc)[:600]],
                "api_error": str(exc)[:600],
            }
            attempts_log.append(err_summary)
            final_result = err_summary
            prior_violations = [str(exc)[:300]]

    return {
        "scenario_id": scenario.scenario_id,
        "site_name": scenario.constraints.site_name,
        "disruption": scenario.disruption.summary,
        "num_tasks": len(scenario.active_tasks()),
        "target_rfs_day": scenario.constraints.target_rfs_day,
        "unmitigated_rfs_day": unmit_rfs,
        "unmitigated_ld_inr_cr": unmit_ld,
        "model": model_name,
        "total_latency_ms": round(total_latency_ms, 1),
        "final": final_result,
        "attempts": attempts_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fair Pure LLM (Gemini) DC scheduling baseline.")
    parser.add_argument("--project-id", default="sd-proj-382223", help="GCP Project ID")
    parser.add_argument("--location", default="global", help="Vertex AI region (e.g., global or us-central1)")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Vertex AI Gemini model ID")
    parser.add_argument(
        "--scenarios",
        default="H01,H02",
        help="Comma-separated held-out scenario IDs (or 'ALL' for H01-H06)",
    )
    parser.add_argument("--retries", type=int, default=1, help="Number of attempts per scenario (1 = zero-shot)")
    args = parser.parse_args()

    all_heldout = evaluator.load_scenarios(split="heldout")
    if args.scenarios.upper() == "ALL":
        selected = all_heldout
    else:
        wanted = {s.strip().upper() for s in args.scenarios.split(",") if s.strip()}
        selected = [sc for sc in all_heldout if sc.scenario_id.upper() in wanted]

    if not selected:
        print(f"No matching held-out scenarios for {args.scenarios}")
        return 1

    print("=" * 76)
    print(f"PURE LLM BASELINE: {args.model} ({args.location}) on {args.project_id}")
    print("=" * 76)

    results: List[Dict[str, Any]] = []
    for sc in selected:
        print(f"\nRunning Pure {args.model} on {sc.scenario_id} ({sc.constraints.site_name}, {len(sc.active_tasks())} tasks)...")
        res = run_llm_on_scenario(sc, args.project_id, args.location, args.model, max_attempts=args.retries)
        results.append(res)
        fin = res["final"]
        verdict = "FEASIBLE" if fin.get("feasible") else f"INFEASIBLE ({fin.get('violation_count')} violations)"
        print(f"  Verdict  : {verdict} | Latency: {res['total_latency_ms']:.0f} ms")
        if fin.get("violations_by_gate"):
            print(f"  By Gate  : {fin['violations_by_gate']}")
        for v in fin.get("sample_violations", [])[:4]:
            print(f"    x {v}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "llm_baseline_result.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "project_id": args.project_id,
                "location": args.location,
                "model": args.model,
                "retries": args.retries,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved audited Pure LLM results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
