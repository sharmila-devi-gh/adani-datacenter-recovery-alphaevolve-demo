#!/usr/bin/env python3
"""3-Way Benchmark Report: Pure LLM vs. Gen-0 Baseline vs. AlphaEvolve Solver.

Evaluates all available solvers on the 6 unseen held-out Data Centre scenarios
(H01-H06) through the identical 5-gate deterministic `verifier.py`, prints a
boardroom-ready comparison table, and writes `artifacts/benchmark_report.json`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiment"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluator
from schema import Scenario
from verifier import compute_unmitigated_rfs_and_ld, verify_plan


def _load_solver_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load solver module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_deterministic_solver(solver_mod: Any, scenario: Scenario) -> Dict[str, Any]:
    unmit_rfs, unmit_ld = compute_unmitigated_rfs_and_ld(scenario)
    t0 = time.perf_counter()
    plans = solver_mod.generate_recovery_plans(scenario, max_plans=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    audited_plans: List[Dict[str, Any]] = []
    norm_pts = []
    all_feasible = bool(plans)
    for p in plans:
        pm = verify_plan(scenario, p, unmitigated_rfs=unmit_rfs, unmitigated_ld=unmit_ld)
        if not pm.feasible:
            all_feasible = False
        else:
            norm_pts.append(evaluator._normalize_objectives(scenario, pm, unmit_rfs, unmit_ld))
        audited_plans.append(
            {
                "plan_label": pm.plan_label,
                "feasible": pm.feasible,
                "rfs_day": pm.rfs_day,
                "rfs_slip_days": pm.rfs_slip_days,
                "rfs_days_recovered": pm.rfs_days_recovered,
                "incremental_cost_inr_cr": pm.incremental_cost_inr_cr,
                "ld_cost_inr_cr": pm.ld_cost_inr_cr,
                "ld_avoided_inr_cr": pm.ld_avoided_inr_cr,
                "net_total_cost_inr_cr": pm.net_total_cost_inr_cr,
                "mean_churn": pm.mean_churn,
                "violations": [f"[{v.gate}] {v.message}" for v in pm.violations],
            }
        )

    hv = evaluator.compute_hypervolume_4d(norm_pts) if all_feasible else 0.0
    best_rfs = max((p["rfs_days_recovered"] for p in audited_plans if p["feasible"]), default=0)
    best_ld = max((p["ld_avoided_inr_cr"] for p in audited_plans if p["feasible"]), default=0.0)
    min_net_cost = min((p["net_total_cost_inr_cr"] for p in audited_plans if p["feasible"]), default=None)

    return {
        "feasible": all_feasible,
        "latency_ms": round(elapsed_ms, 2),
        "hypervolume_4d": round(hv, 4),
        "num_plans": len(audited_plans),
        "pareto_count": len(evaluator._pareto_filter(norm_pts)) if all_feasible else 0,
        "best_rfs_days_recovered": best_rfs,
        "best_ld_avoided_inr_cr": round(best_ld, 2),
        "min_net_total_cost_inr_cr": round(min_net_cost, 2) if min_net_cost is not None else None,
        "plans": audited_plans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pure LLM vs. Gen-0 Baseline vs. AlphaEvolve on held-out DC scenarios.")
    parser.add_argument(
        "--evolved-solver",
        default=str(ARTIFACTS_DIR / "evolved_solver.py"),
        help="Path to AlphaEvolve evolved_solver.py",
    )
    parser.add_argument(
        "--llm-results",
        default=str(ARTIFACTS_DIR / "llm_baseline_result.json"),
        help="Path to Pure LLM baseline JSON produced by scripts/run_llm_baseline.py",
    )
    args = parser.parse_args()

    heldout = evaluator.load_scenarios(split="heldout")
    baseline_mod = _load_solver_from_path(EXP_DIR / "solver.py", "baseline_solver_mod")

    evolved_path = Path(args.evolved_solver)
    evolved_mod = _load_solver_from_path(evolved_path, "evolved_solver_mod") if evolved_path.is_file() else None

    llm_by_id: Dict[str, Dict[str, Any]] = {}
    llm_path = Path(args.llm_results)
    if llm_path.is_file():
        llm_blob = json.loads(llm_path.read_text(encoding="utf-8"))
        for row in llm_blob.get("results", []):
            llm_by_id[row["scenario_id"]] = row

    print("=" * 102)
    print("3-WAY HELD-OUT BENCHMARK: Pure LLM vs. Gen-0 Baseline vs. AlphaEvolve Solver (Unseen DC Sites H01-H06)")
    print("=" * 102)
    header = (
        f"{'Site':<5} | {'Pure LLM':<24} | {'Gen-0 Baseline':<28} | {'AlphaEvolve Evolved':<32}"
    )
    print(header)
    print("-" * 102)

    scenario_reports: List[Dict[str, Any]] = []
    for sc in heldout:
        unmit_rfs, unmit_ld = compute_unmitigated_rfs_and_ld(sc)
        base_res = _run_deterministic_solver(baseline_mod, sc)
        evo_res = _run_deterministic_solver(evolved_mod, sc) if evolved_mod is not None else None
        llm_row = llm_by_id.get(sc.scenario_id)

        if llm_row:
            fin = llm_row["final"]
            llm_str = (
                f"OK {fin.get('rfs_days_recovered', 0)}d ({llm_row['total_latency_ms']/1000:.1f}s)"
                if fin.get("feasible")
                else f"FAIL {fin.get('violation_count')}v ({llm_row['total_latency_ms']/1000:.1f}s)"
            )
        else:
            llm_str = "(run_llm_baseline.py)"

        base_str = (
            f"OK HV={base_res['hypervolume_4d']:.3f} +{base_res['best_rfs_days_recovered']}d "
            f"({base_res['latency_ms']:.1f}ms)"
        )
        if evo_res:
            evo_str = (
                f"{'OK' if evo_res['feasible'] else 'FAIL'} HV={evo_res['hypervolume_4d']:.3f} "
                f"+{evo_res['best_rfs_days_recovered']}d P={evo_res['pareto_count']} ({evo_res['latency_ms']:.1f}ms)"
            )
        else:
            evo_str = "(run_evolution.py first)"

        print(f"{sc.scenario_id:<5} | {llm_str:<24} | {base_str:<28} | {evo_str:<32}")
        scenario_reports.append(
            {
                "scenario_id": sc.scenario_id,
                "site_name": sc.constraints.site_name,
                "disruption": sc.disruption.summary,
                "target_rfs_day": sc.constraints.target_rfs_day,
                "unmitigated_rfs_day": unmit_rfs,
                "unmitigated_ld_inr_cr": unmit_ld,
                "pure_llm": llm_row,
                "gen0_baseline": base_res,
                "alphaevolve": evo_res,
            }
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "benchmark_report.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "has_evolved_solver": evolved_mod is not None,
                "has_llm_results": bool(llm_by_id),
                "scenarios": scenario_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 102)
    print(f"Saved full 3-way benchmark report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
