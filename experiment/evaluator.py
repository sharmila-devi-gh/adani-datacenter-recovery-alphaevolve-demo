"""Client-side multi-objective AlphaEvolve evaluator for DC schedule recovery.

Design principles:
  1. Privacy-by-design: Train & held-out JSON scenarios reside strictly inside
     local/tenant `experiment/data/` and are never sent to the AlphaEvolve API.
  2. Strict maximization semantics: `pareto_hypervolume_score` is maximized;
     minimization KPIs (cost, churn, runtime) are negated.
  3. Clean failure contract: Any crash, timeout, empty output, NaN/inf, or hard
     feasibility violation returns `score=None` (never `0.0`) with actionable
     diagnostic insights so AlphaEvolve self-corrects in the next generation.
  4. Dual compatibility: Uses `alpha_evolve.models` when running inside Google
     Cloud Cluster Toolkit / Cloud Batch, and falls back seamlessly to identical
     plain dicts when running standalone in a local Python environment.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure local experiment modules resolve regardless of working directory
_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from generator import generate_benchmark_suite
from schema import PlanMetrics, RecoveryPlan, Scenario, scenario_from_dict
from verifier import compute_unmitigated_rfs_and_ld, verify_plan

try:
    from alpha_evolve.models import (
        AlphaEvolveEvaluationInsight,
        AlphaEvolveEvaluationInsights,
        AlphaEvolveEvaluationScore,
        AlphaEvolveEvaluationScores,
        AlphaEvolveProgramEvaluation,
    )

    _HAS_AE_MODELS = True
except ImportError:
    _HAS_AE_MODELS = False

PRIMARY_METRIC = "pareto_hypervolume_score"
ALL_METRICS = (
    PRIMARY_METRIC,
    "mean_rfs_days_recovered",
    "neg_mean_cost_inr_cr",
    "mean_ld_avoided_inr_cr",
    "neg_mean_churn",
    "neg_p95_runtime_ms",
)
SCENARIO_TIMEOUT_SEC = 2.0
SUITE_TIMEOUT_SEC = 30


class EvaluationTimeoutError(TimeoutError):
    """Raised when a candidate solver exceeds the wall-clock evaluation guard."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise EvaluationTimeoutError(f"Candidate evaluation exceeded {SUITE_TIMEOUT_SEC}s wall-clock limit")


def _format_evaluation(
    scores_map: Dict[str, Optional[float]],
    insights: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Build exact AlphaEvolveProgramEvaluation dict (Pydantic-validated if available)."""
    sanitized_scores: Dict[str, Optional[float]] = {}
    for k, v in scores_map.items():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            sanitized_scores[k] = None
        else:
            sanitized_scores[k] = round(float(v), 6)

    if _HAS_AE_MODELS:
        return AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(
                scores=[
                    AlphaEvolveEvaluationScore(metric=m, score=sanitized_scores.get(m))
                    for m in ALL_METRICS
                ]
            ),
            insights=AlphaEvolveEvaluationInsights(
                insights=[
                    AlphaEvolveEvaluationInsight(label=lbl, text=txt[:1000])
                    for lbl, txt in insights
                ]
            ),
        ).model_dump()

    return {
        "scores": {
            "scores": [
                {"metric": m, "score": sanitized_scores.get(m)} for m in ALL_METRICS
            ]
        },
        "insights": {
            "insights": [
                {"label": lbl, "text": txt[:1000]} for lbl, txt in insights
            ]
        },
    }


def load_scenarios(split: str = "train", data_root: Optional[Path] = None) -> List[Scenario]:
    """Load synthetic DC scenarios from disk, generating them deterministically if missing."""
    root = data_root or (_EXP_DIR / "data")
    sub = root / ("train_scenarios" if split == "train" else "heldout_scenarios")
    if not sub.exists() or not list(sub.glob("*.json")):
        generate_benchmark_suite(root)
    files = sorted(sub.glob("*.json"))
    return [scenario_from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]


def _normalize_objectives(
    scenario: Scenario,
    metrics: PlanMetrics,
    unmitigated_rfs: int,
    unmitigated_ld: float,
) -> Tuple[float, float, float, float]:
    """Map a verified feasible plan onto 4 maximized objectives in [0, 1]."""
    max_recoverable = max(1, unmitigated_rfs - scenario.constraints.target_rfs_day)
    m_rfs = max(0.0, min(1.0, metrics.rfs_days_recovered / float(max_recoverable)))

    max_budget = max(0.1, scenario.constraints.max_recovery_budget_inr_cr)
    m_cost = max(0.0, min(1.0, 1.0 - (metrics.incremental_cost_inr_cr / max_budget)))

    if unmitigated_ld > 0.01:
        m_ld = max(0.0, min(1.0, metrics.ld_avoided_inr_cr / unmitigated_ld))
    else:
        m_ld = 1.0 if metrics.ld_cost_inr_cr == 0.0 else 0.0

    m_churn = max(0.0, min(1.0, 1.0 - metrics.mean_churn))
    return (m_rfs, m_cost, m_ld, m_churn)


def _pareto_filter(points: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """Return non-dominated 4D points (all objectives maximized)."""
    unique = list(dict.fromkeys(points))
    frontier: List[Tuple[float, float, float, float]] = []
    for i, p in enumerate(unique):
        dominated = False
        for j, q in enumerate(unique):
            if i == j:
                continue
            if all(q[k] >= p[k] for k in range(4)) and any(q[k] > p[k] for k in range(4)):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def compute_hypervolume_4d(
    points: Sequence[Tuple[float, float, float, float]],
    n_samples: int = 2000,
) -> float:
    """Deterministic 4D dominated hypervolume estimate in [0, 1]^4 with ref=(0,0,0,0)."""
    frontier = _pareto_filter(points)
    if not frontier:
        return 0.0
    # Fixed deterministic Sobol-like Halton/LCG sampling for zero variance across evaluations
    rng = random.Random(20261125)
    dominated_hits = 0
    for _ in range(n_samples):
        u0, u1, u2, u3 = rng.random(), rng.random(), rng.random(), rng.random()
        for p0, p1, p2, p3 in frontier:
            if p0 >= u0 and p1 >= u1 and p2 >= u2 and p3 >= u3:
                dominated_hits += 1
                break
    return dominated_hits / float(n_samples)


def evaluate_dc_recovery(
    solver_module: Any = None,
    split: str = "train",
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Official AlphaEvolve entrypoint: evaluates `solver.generate_recovery_plans`."""
    old_handler = None
    has_sigalrm = hasattr(signal, "SIGALRM")
    if has_sigalrm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(SUITE_TIMEOUT_SEC)

    try:
        if solver_module is None:
            import solver as default_solver  # type: ignore

            solver_module = importlib.reload(default_solver)

        if not hasattr(solver_module, "generate_recovery_plans"):
            return _format_evaluation(
                {m: None for m in ALL_METRICS},
                [("interface_error", "solver.py must define generate_recovery_plans(scenario, max_plans=5)")],
            )

        scenarios = load_scenarios(split=split, data_root=data_root)
        hvs: List[float] = []
        best_rfs_list: List[float] = []
        mean_cost_list: List[float] = []
        best_ld_list: List[float] = []
        mean_churn_list: List[float] = []
        runtimes_ms: List[float] = []
        frontier_sizes: List[int] = []

        for sc in scenarios:
            unmit_rfs, unmit_ld = compute_unmitigated_rfs_and_ld(sc)
            t0 = time.perf_counter()
            plans = solver_module.generate_recovery_plans(sc, max_plans=5)
            elapsed_s = time.perf_counter() - t0
            runtimes_ms.append(elapsed_s * 1000.0)

            if elapsed_s > SCENARIO_TIMEOUT_SEC:
                return _format_evaluation(
                    {m: None for m in ALL_METRICS},
                    [
                        (
                            "latency_violation",
                            f"Scenario {sc.scenario_id} took {elapsed_s:.2f}s (> {SCENARIO_TIMEOUT_SEC:.1f}s per-scenario cap). Optimize inner loops.",
                        )
                    ],
                )

            if not isinstance(plans, list) or len(plans) == 0:
                return _format_evaluation(
                    {m: None for m in ALL_METRICS},
                    [("output_error", f"Scenario {sc.scenario_id}: returned empty or non-list plans.")],
                )
            if len(plans) > 5:
                return _format_evaluation(
                    {m: None for m in ALL_METRICS},
                    [("output_error", f"Scenario {sc.scenario_id}: returned {len(plans)} plans (max 5 allowed).")],
                )

            norm_pts: List[Tuple[float, float, float, float]] = []
            plan_metrics_list: List[PlanMetrics] = []
            for idx, plan in enumerate(plans):
                if not isinstance(plan, RecoveryPlan):
                    return _format_evaluation(
                        {m: None for m in ALL_METRICS},
                        [("type_error", f"Scenario {sc.scenario_id}: plan[{idx}] is not a RecoveryPlan instance.")],
                    )
                pm = verify_plan(sc, plan, unmitigated_rfs=unmit_rfs, unmitigated_ld=unmit_ld)
                if not pm.feasible:
                    v_summary = "; ".join(f"[{v.gate}] {v.message}" for v in pm.violations[:3])
                    return _format_evaluation(
                        {m: None for m in ALL_METRICS},
                        [
                            (
                                "feasibility_violation",
                                f"Scenario {sc.scenario_id} ({sc.disruption.scenario_type}), plan '{plan.plan_label}' failed {len(pm.violations)} hard gate(s): {v_summary}",
                            )
                        ],
                    )
                plan_metrics_list.append(pm)
                norm_pts.append(_normalize_objectives(sc, pm, unmit_rfs, unmit_ld))

            frontier = _pareto_filter(norm_pts)
            hv = compute_hypervolume_4d(frontier)
            hvs.append(hv)
            frontier_sizes.append(len(frontier))
            best_rfs_list.append(float(max(pm.rfs_days_recovered for pm in plan_metrics_list)))
            mean_cost_list.append(
                sum(pm.incremental_cost_inr_cr for pm in plan_metrics_list) / len(plan_metrics_list)
            )
            best_ld_list.append(float(max(pm.ld_avoided_inr_cr for pm in plan_metrics_list)))
            mean_churn_list.append(
                sum(pm.mean_churn for pm in plan_metrics_list) / len(plan_metrics_list)
            )

        mean_hv = sum(hvs) / len(hvs)
        mean_runtime_s = (sum(runtimes_ms) / len(runtimes_ms)) / 1000.0
        sorted_rt = sorted(runtimes_ms)
        p95_rt_ms = sorted_rt[min(len(sorted_rt) - 1, int(math.ceil(0.95 * len(sorted_rt))) - 1)]

        # Small tie-breaker bonus for fast sub-second solvers + diversity reward
        avg_frontier = sum(frontier_sizes) / len(frontier_sizes)
        speed_bonus = 0.015 * max(0.0, 1.0 - (mean_runtime_s / SCENARIO_TIMEOUT_SEC))
        diversity_bonus = 0.015 * min(1.0, avg_frontier / 4.0)
        primary_score = mean_hv + speed_bonus + diversity_bonus

        mean_rfs = sum(best_rfs_list) / len(best_rfs_list)
        mean_cost = sum(mean_cost_list) / len(mean_cost_list)
        mean_ld = sum(best_ld_list) / len(best_ld_list)
        mean_churn = sum(mean_churn_list) / len(mean_churn_list)

        # Actionable optimization hint identifying the weakest scenario
        worst_idx = min(range(len(hvs)), key=lambda i: hvs[i])
        worst_sc = scenarios[worst_idx]
        insight_text = (
            f"All {len(scenarios)} {split} scenarios 100% feasible! "
            f"Mean 4D HV={mean_hv:.4f}, avg Pareto size={avg_frontier:.1f}/5, "
            f"best RFS recovered={mean_rfs:.1f}d, best LD avoided=INR {mean_ld:.2f} Cr, "
            f"mean cost=INR {mean_cost:.2f} Cr, P95 latency={p95_rt_ms:.1f}ms. "
            f"Lowest HV scenario: {worst_sc.scenario_id} ({worst_sc.disruption.scenario_type}, HV={hvs[worst_idx]:.4f}). "
            f"Tip: use ALT_SUBCONTRACTOR selectively on Kronos-bottlenecked days and local-search critical tasks across 5 diverse cost/RFS/churn weights."
        )

        return _format_evaluation(
            {
                PRIMARY_METRIC: primary_score,
                "mean_rfs_days_recovered": mean_rfs,
                "neg_mean_cost_inr_cr": -mean_cost,
                "mean_ld_avoided_inr_cr": mean_ld,
                "neg_mean_churn": -mean_churn,
                "neg_p95_runtime_ms": -p95_rt_ms,
            },
            [("evaluation_summary", insight_text)],
        )

    except EvaluationTimeoutError as exc:
        return _format_evaluation(
            {m: None for m in ALL_METRICS},
            [("timeout_error", str(exc))],
        )
    except Exception as exc:
        return _format_evaluation(
            {m: None for m in ALL_METRICS},
            [("runtime_error", f"{type(exc).__name__}: {exc}")],
        )
    finally:
        if has_sigalrm and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def evaluate_candidate_files_isolated(
    candidate_files: Dict[str, str],
    split: str = "train",
    timeout_sec: int = SUITE_TIMEOUT_SEC + 5,
) -> Dict[str, Any]:
    """Execute untrusted AlphaEvolve candidate code in an isolated local subprocess.

    Demonstrates the client-side verification sandbox: only candidate source code
    enters; only the JSON scalar evaluation dict exits.
    """
    with tempfile.TemporaryDirectory(prefix="ae_eval_sandbox_") as tmpdir:
        tmp = Path(tmpdir)
        # Always preserve immutable platform files from _EXP_DIR
        for immutable in ("schema.py", "verifier.py", "generator.py", "evaluator.py"):
            (tmp / immutable).write_text((_EXP_DIR / immutable).read_text(encoding="utf-8"), encoding="utf-8")

        # Write candidate mutated file(s) — ONLY allow solver.py to be overwritten!
        if "solver.py" not in candidate_files:
            return _format_evaluation(
                {m: None for m in ALL_METRICS},
                [("candidate_error", "Candidate payload missing 'solver.py'")],
            )
        (tmp / "solver.py").write_text(candidate_files["solver.py"], encoding="utf-8")

        data_root = (_EXP_DIR / "data").resolve()
        runner_code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(tmp)!r})\n"
            "import evaluator\n"
            f"res = evaluator.evaluate_dc_recovery(split={split!r}, data_root=Path({str(data_root)!r}))\n"
            "print('__AE_RESULT_START__')\n"
            "print(json.dumps(res))\n"
            "print('__AE_RESULT_END__')\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner_code],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(tmp),
            )
        except subprocess.TimeoutExpired:
            return _format_evaluation(
                {m: None for m in ALL_METRICS},
                [("timeout_error", f"Isolated subprocess timed out after {timeout_sec}s")],
            )

        out = proc.stdout
        if "__AE_RESULT_START__" in out and "__AE_RESULT_END__" in out:
            payload = out.split("__AE_RESULT_START__", 1)[1].split("__AE_RESULT_END__", 1)[0].strip()
            return json.loads(payload)

        err_msg = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        return _format_evaluation(
            {m: None for m in ALL_METRICS},
            [("sandbox_crash", f"Candidate crashed in sandbox: {err_msg[-800:]}")],
        )


if __name__ == "__main__":
    result = evaluate_dc_recovery(split="train")
    print(json.dumps(result, indent=2))
