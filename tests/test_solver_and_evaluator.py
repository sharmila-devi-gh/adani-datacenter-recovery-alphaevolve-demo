"""Unit tests for Gen-0 solver, EVOLVE-BLOCK lint, and sandboxed evaluator contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiment"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluator
import run_evolution


class TestSolverAndEvaluator(unittest.TestCase):
    def test_evolve_block_markers_are_strict_column_zero(self) -> None:
        code = (REPO_ROOT / "experiment" / "solver.py").read_text(encoding="utf-8")
        run_evolution.validate_evolve_block_markers(code, "solver.py")

    def test_gen0_seed_scores_cleanly_in_isolated_sandbox(self) -> None:
        code = (REPO_ROOT / "experiment" / "solver.py").read_text(encoding="utf-8")
        res = evaluator.evaluate_candidate_files_isolated({"solver.py": code}, split="train")
        primary = run_evolution.extract_primary_score(res)
        self.assertIsNotNone(primary)
        self.assertGreater(primary, 0.15)
        self.assertLess(primary, 0.85)  # Leaves clear headroom for AlphaEvolve

    def test_invalid_candidate_returns_score_none_not_zero(self) -> None:
        broken_code = """
from schema import RecoveryPlan, TaskAssignment
def generate_recovery_plans(scenario, max_plans=5):
    # Intentionally schedule every task at day 0 -> guaranteed hard violations!
    asgns = {}
    for t in scenario.active_tasks():
        m = t.available_modes[0]
        asgns[t.task_id] = TaskAssignment(t.task_id, 0, m.duration_days, m.mode)
    return [RecoveryPlan("Broken", asgns)]
"""
        res = evaluator.evaluate_candidate_files_isolated({"solver.py": broken_code}, split="train")
        primary = run_evolution.extract_primary_score(res)
        self.assertIsNone(primary, "Invalid candidate MUST return score=None, never 0.0")
        insight = run_evolution.extract_first_insight(res)
        self.assertIn("feasibility_violation", insight)

    def test_crashing_candidate_returns_score_none_with_diagnostics(self) -> None:
        crashing_code = "def generate_recovery_plans(scenario, max_plans=5):\n    raise ZeroDivisionError('boom')\n"
        res = evaluator.evaluate_candidate_files_isolated({"solver.py": crashing_code}, split="train")
        self.assertIsNone(run_evolution.extract_primary_score(res))
        self.assertIn("ZeroDivisionError", run_evolution.extract_first_insight(res))

    def test_timeout_candidate_returns_score_none(self) -> None:
        infinite_code = "import time\ndef generate_recovery_plans(scenario, max_plans=5):\n    time.sleep(10)\n    return []\n"
        res = evaluator.evaluate_candidate_files_isolated(
            {"solver.py": infinite_code}, split="train", timeout_sec=3
        )
        self.assertIsNone(run_evolution.extract_primary_score(res))
        self.assertIn("timeout", run_evolution.extract_first_insight(res).lower())


if __name__ == "__main__":
    unittest.main()
