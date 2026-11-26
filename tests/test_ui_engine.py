"""Unit tests for the CXO Streamlit UI data engine and preview reference solver."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiment"))

import evaluator
import reference_evolved_solver
import solver as gen0_solver
from ui import demo_engine


class TestDemoUiEngine(unittest.TestCase):
    def test_reference_solver_passes_full_train_suite_under_latency_cap(self) -> None:
        res = evaluator.evaluate_dc_recovery(reference_evolved_solver, split="train")
        scores = {s["metric"]: s["score"] for s in res["scores"]["scores"]}
        self.assertIsNotNone(scores["pareto_hypervolume_score"], f"Failed: {res['insights']}")
        gen0_res = evaluator.evaluate_dc_recovery(gen0_solver, split="train")
        gen0_hv = {s["metric"]: s["score"] for s in gen0_res["scores"]["scores"]}["pareto_hypervolume_score"]
        self.assertGreater(scores["pareto_hypervolume_score"], gen0_hv)

    def test_heldout_hero_h02_shows_clear_lift_and_zero_violations(self) -> None:
        h02 = next(sc for sc in demo_engine.load_heldout_scenarios() if sc.scenario_id == "H02")
        base_res = demo_engine.evaluate_solver_on_scenario(gen0_solver, h02)
        ref_res = demo_engine.evaluate_solver_on_scenario(reference_evolved_solver, h02)
        self.assertTrue(ref_res["feasible"])
        self.assertGreater(ref_res["hypervolume_4d"], base_res["hypervolume_4d"])
        self.assertGreater(
            ref_res["best_rfs_plan"]["rfs_days_recovered"],
            base_res["best_rfs_plan"]["rfs_days_recovered"],
        )

    def test_what_if_sliders_and_gantt_and_kronos_profile(self) -> None:
        h01 = next(sc for sc in demo_engine.load_heldout_scenarios() if sc.scenario_id == "H01")
        stressed = demo_engine.apply_what_if_overrides(
            h01,
            extra_fim_delay_days=15,
            cost_shock_pct_override=35,
            budget_cap_override_cr=24.0,
        )
        self.assertEqual(stressed.constraints.max_recovery_budget_inr_cr, 24.0)
        self.assertAlmostEqual(stressed.disruption.cost_shock_multiplier, 1.35)

        ae_res = demo_engine.evaluate_solver_on_scenario(reference_evolved_solver, stressed)
        self.assertTrue(ae_res["feasible"])
        chosen_plan = ae_res["best_rfs_plan"]["plan"]

        timeline = demo_engine.compute_wbs_stage_timeline(stressed, chosen_plan)
        self.assertEqual(len(timeline), 5)

        kronos = demo_engine.compute_kronos_daily_profile(stressed, chosen_plan, trade="MEP_HVAC")
        self.assertTrue(len(kronos) > 0)
        # Every single day MUST obey Kronos daily cap!
        for row in kronos:
            self.assertLessEqual(row["scheduled_crew"], row["kronos_cap"])

    def test_pure_llm_preview_audit_surfaces_real_verifier_violations(self) -> None:
        h01 = next(sc for sc in demo_engine.load_heldout_scenarios() if sc.scenario_id == "H01")
        llm = demo_engine.load_or_simulate_pure_llm_audit(h01)
        self.assertFalse(llm["feasible"])
        self.assertIsNone(llm["latency_ms"])
        self.assertGreater(llm["violation_count"], 0)
        self.assertTrue(len(llm["sample_violations"]) > 0)

    def test_tight_budget_cap_reduces_recovered_days_on_h02(self) -> None:
        h02 = next(sc for sc in demo_engine.load_heldout_scenarios() if sc.scenario_id == "H02")
        full_res = demo_engine.evaluate_solver_on_scenario(reference_evolved_solver, h02)
        tight_sc = demo_engine.apply_what_if_overrides(h02, budget_cap_override_cr=4.0)
        tight_res = demo_engine.evaluate_solver_on_scenario(reference_evolved_solver, tight_sc)
        self.assertTrue(tight_res["feasible"])
        self.assertLess(
            tight_res["best_rfs_plan"]["rfs_days_recovered"],
            full_res["best_rfs_plan"]["rfs_days_recovered"],
        )

    def test_privacy_and_lineage_loader_reports_zero_data_egress(self) -> None:
        info = demo_engine.load_privacy_and_lineage_artifacts()
        self.assertEqual(info["raw_schedule_bytes_egressed"], 0)
        self.assertTrue(len(info["diff_text"]) > 50)

    def test_live_demo_change_order_gate_certificate_and_p6_csv_export(self) -> None:
        h02 = next(sc for sc in demo_engine.load_heldout_scenarios() if sc.scenario_id == "H02")
        unmit_stages = demo_engine.compute_unmitigated_stage_timeline(h02)
        self.assertEqual(len(unmit_stages), 5)

        ae_res = demo_engine.evaluate_solver_on_scenario(reference_evolved_solver, h02)
        best_row = ae_res["best_roi_plan"]
        changed_tasks = demo_engine.compute_modified_task_table(h02, best_row["plan"])
        self.assertGreater(len(changed_tasks), 0)
        self.assertTrue(any(r["Action Mode"] != "NORMAL" for r in changed_tasks))

        gates = demo_engine.compute_gate_check_summary(h02, best_row)
        self.assertEqual(len(gates), 5)
        self.assertTrue(all(g["Status"] == "PASS" for g in gates))

        csv_text = demo_engine.export_plan_to_p6_csv(h02, best_row["plan"])
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        self.assertEqual(len(lines), len(h02.active_tasks()) + 1)


if __name__ == "__main__":
    unittest.main()
