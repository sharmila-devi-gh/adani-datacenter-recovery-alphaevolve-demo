"""Unit tests for AlphaEvolve REST payload schemas and multi-file unflattening."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiment"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import alphaevolve_client
import evaluator
import run_evolution
import run_llm_baseline


class TestAlphaEvolveClientPayloads(unittest.TestCase):
    def test_build_experiment_payloads_match_api_contract(self) -> None:
        seed_code = (REPO_ROOT / "experiment" / "solver.py").read_text(encoding="utf-8")
        seed_eval = evaluator.evaluate_candidate_files_isolated({"solver.py": seed_code}, split="train")
        exp_cfg, init_prog = run_evolution.build_experiment_payloads(
            max_programs=5,
            concurrency=1,
            model_spec="gemini-3.8-flash:0.6;gemini-3.1-pro-preview:0.4",
            seed_eval=seed_eval,
        )
        self.assertEqual(exp_cfg["program_language"], "python")
        self.assertEqual(exp_cfg["run_settings"]["max_duration"], "7200s")
        self.assertEqual(exp_cfg["run_settings"]["idle_timeout"], "3600s")
        self.assertTrue(exp_cfg["generation_settings"]["include_full_program_in_prompt"])
        self.assertEqual(len(exp_cfg["generation_settings"]["models"]), 2)
        self.assertEqual(
            exp_cfg["evolution_settings"]["parent_sampling_config"]["pareto_sampling_config"][
                "pareto_sampling_probability"
            ],
            0.35,
        )
        # Privacy check: zero schedule filenames or synthetic DC site names inside initial_program payload
        raw_payload = json.dumps(init_prog)
        self.assertNotIn("Navi Mumbai DC-2", raw_payload)
        self.assertNotIn("train_scenarios", raw_payload)
        paths = [f["path"] for f in init_prog["content"]["files"]]
        self.assertEqual(paths, ["solver.py", "schema.py"])

    def test_fix_multi_files_program_unflattens_wire_response(self) -> None:
        flat = (
            "File: solver.py\nLanguage: python\n---\nprint('evolved')\n---\n"
            "File: schema.py\nLanguage: python\n---\nX = 1\n---"
        )
        prog = {"name": "p1", "content": {"files": [{"path": "combined", "content": flat}]}}
        alphaevolve_client.fix_multi_files_program(prog)
        files = {f["path"]: f["content"] for f in prog["content"]["files"]}
        self.assertEqual(files["solver.py"], "print('evolved')")
        self.assertEqual(files["schema.py"], "X = 1")

    def test_sanitize_relative_path_blocks_traversal(self) -> None:
        with self.assertRaises(ValueError):
            alphaevolve_client.sanitize_relative_path("../../etc/passwd")

    def test_fair_llm_prompt_includes_all_hard_constraints(self) -> None:
        h01 = next(sc for sc in evaluator.load_scenarios(split="heldout") if sc.scenario_id == "H01")
        prompt = run_llm_baseline.build_fair_llm_prompt(h01)
        self.assertIn("kronos_daily_cap", prompt)
        self.assertIn("effective_fim_arrival_day", prompt)
        self.assertIn("zone_max_concurrent", prompt)
        self.assertIn("max_recovery_budget_inr_cr", prompt)


if __name__ == "__main__":
    unittest.main()
