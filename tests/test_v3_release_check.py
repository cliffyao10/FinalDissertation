import json
import tempfile
import unittest
from pathlib import Path

from recommendation_training.v3_release_check import REQUIRED, evaluate_gate


class V3ReleaseCheckTests(unittest.TestCase):
    def test_missing_artifacts_block_release(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_gate(directory)
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision"], "v3_incomplete")

    def test_complete_evidence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in REQUIRED:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            comparison = {
                "shared_head_metrics_v2_and_v3": {
                    "auc": 0.74,
                    "auc_95_percent_ci": [0.71, 0.78],
                },
                "search_ablation_same_abstract_candidate_space": {
                    "deterministic_repeat_top10": True,
                    "model_score_regret_quota": {"minimum": 0.0},
                },
                "runtime_seconds_same_machine": {
                    "v3_exact": {"median": 0.5, "p95": 2.0}
                },
                "candidate_spaces": {
                    "v3_abstract": {
                        "privacy_contract": {
                            "all_abstract": True,
                            "nonempty_image_paths": 0,
                            "records_with_forbidden_commercial_fields": 0,
                        }
                    }
                },
                "weather_style_coverage": {
                    "cases": 36,
                    "all_cases_have_four_logical_slots": True,
                },
                "research_candidate_gate": {"promotion_allowed": True},
            }
            (root / "results/v3_system_comparison.json").write_text(
                json.dumps(comparison), encoding="utf-8"
            )
            (root / "results/d2_evaluation.json").write_text(
                json.dumps({"passed": True, "gates": {"fitb": True}}),
                encoding="utf-8",
            )
            report = evaluate_gate(root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "v3_release_ready")


if __name__ == "__main__":
    unittest.main()
