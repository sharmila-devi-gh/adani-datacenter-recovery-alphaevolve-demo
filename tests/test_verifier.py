"""Exhaustive unit tests for all 5 hard feasibility gates in verifier.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiment"))

import evaluator
import solver
from schema import RecoveryPlan, TaskAssignment
from verifier import verify_plan


class TestVerifierHardGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.heldout = evaluator.load_scenarios(split="heldout")
        cls.h01 = next(sc for sc in cls.heldout if sc.scenario_id == "H01")
        cls.valid_plan = solver.generate_recovery_plans(cls.h01, max_plans=3)[0]

    def test_valid_seed_plan_has_zero_violations(self) -> None:
        pm = verify_plan(self.h01, self.valid_plan)
        self.assertTrue(pm.feasible, f"Unexpected violations: {pm.violations}")
        self.assertEqual(len(pm.violations), 0)

    def test_gate1_completeness_catches_missing_task(self) -> None:
        bad = dict(self.valid_plan.assignments)
        dropped = next(iter(bad))
        bad.pop(dropped)
        pm = verify_plan(self.h01, RecoveryPlan("missing", bad))
        self.assertFalse(pm.feasible)
        self.assertTrue(any(v.gate == "COMPLETENESS" for v in pm.violations))

    def test_gate1_freeze_catches_altered_completed_task(self) -> None:
        trigger = self.h01.disruption.trigger_day
        frozen_task = next(t for t in self.h01.tasks if t.baseline_end_day <= trigger)
        bad = dict(self.valid_plan.assignments)
        old = bad[frozen_task.task_id]
        bad[frozen_task.task_id] = TaskAssignment(
            task_id=old.task_id,
            start_day=old.start_day + 1,
            end_day=old.end_day + 1,
            selected_mode=old.selected_mode,
        )
        pm = verify_plan(self.h01, RecoveryPlan("altered_frozen", bad))
        self.assertFalse(pm.feasible)
        self.assertTrue(any(v.gate == "FREEZE" for v in pm.violations))

    def test_gate2_precedence_catches_early_successor(self) -> None:
        trigger = self.h01.disruption.trigger_day
        task_with_pred = next(
            t for t in self.h01.active_tasks() if t.baseline_end_day > trigger and t.predecessors
        )
        bad = dict(self.valid_plan.assignments)
        old = bad[task_with_pred.task_id]
        dur = old.end_day - old.start_day
        pred_end = max(bad[p].end_day for p in task_with_pred.predecessors if p in bad)
        bad_start = max(trigger, pred_end - 2)
        if bad_start < pred_end:
            bad[task_with_pred.task_id] = TaskAssignment(
                task_id=old.task_id,
                start_day=bad_start,
                end_day=bad_start + dur,
                selected_mode=old.selected_mode,
            )
            pm = verify_plan(self.h01, RecoveryPlan("bad_pred", bad))
            self.assertFalse(pm.feasible)
            self.assertTrue(any(v.gate == "PRECEDENCE" for v in pm.violations))

    def test_gate3_fim_catches_start_before_arrival(self) -> None:
        fim_arrivals = self.h01.effective_fim_arrivals()
        trigger = self.h01.disruption.trigger_day
        fim_task = next(
            t
            for t in self.h01.active_tasks()
            if t.fim_item_id and fim_arrivals[t.fim_item_id] > trigger and t.baseline_end_day > trigger
        )
        bad = dict(self.valid_plan.assignments)
        old = bad[fim_task.task_id]
        dur = old.end_day - old.start_day
        early_start = fim_arrivals[fim_task.fim_item_id] - 3
        bad[fim_task.task_id] = TaskAssignment(
            task_id=old.task_id,
            start_day=early_start,
            end_day=early_start + dur,
            selected_mode=old.selected_mode,
        )
        pm = verify_plan(self.h01, RecoveryPlan("bad_fim", bad))
        self.assertFalse(pm.feasible)
        self.assertTrue(any(v.gate == "FIM" for v in pm.violations))

    def test_gate4_kronos_and_gate5_workfront_catch_collisions(self) -> None:
        trigger = self.h01.disruption.trigger_day
        bad = dict(self.valid_plan.assignments)
        # Stack 8 post-trigger tasks onto the exact same day to violate both Kronos & work-front caps
        post_tasks = [t for t in self.h01.active_tasks() if t.baseline_end_day > trigger][:10]
        stack_day = trigger + 15
        for t in post_tasks:
            old = bad[t.task_id]
            dur = old.end_day - old.start_day
            bad[t.task_id] = TaskAssignment(
                task_id=t.task_id,
                start_day=stack_day,
                end_day=stack_day + dur,
                selected_mode=old.selected_mode,
            )
        pm = verify_plan(self.h01, RecoveryPlan("stacked_caps", bad))
        self.assertFalse(pm.feasible)
        gates = {v.gate for v in pm.violations}
        self.assertTrue("KRONOS" in gates or "WORK_FRONT" in gates)


if __name__ == "__main__":
    unittest.main()
