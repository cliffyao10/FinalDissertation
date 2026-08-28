import json
import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.disjoint_release_check import REQUIRED_RESULTS, run_checks


class DisjointReleaseCheckTests(unittest.TestCase):
    def test_missing_evidence_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_checks(directory)
        self.assertFalse(report["production_promotion_allowed"])

    def test_fitb_at_chance_blocks_otherwise_complete_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in REQUIRED_RESULTS.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / REQUIRED_RESULTS["baselines"]).write_text(json.dumps({
                "baselines": {
                    "learned_compatibility_ranker": {"auc_95_percent_ci": [0.85, 0.87]},
                    "mean_pairwise_siglip_cosine": {"auc_95_percent_ci": [0.69, 0.71]},
                }
            }), encoding="utf-8")
            models = {
                name: {"metrics": {"auc": {
                    "mean": value, "available_runs": 3,
                    "sample_standard_deviation": 0.001,
                }}}
                for name, value in {
                    "bilstm": 0.79,
                    "type_aware_pairwise": 0.78,
                    "set_transformer": 0.77,
                    "lightweight_pairwise_proposed": 0.86,
                }.items()
            }
            (root / REQUIRED_RESULTS["architectures"]).write_text(
                json.dumps({"summary": models}), encoding="utf-8"
            )
            (root / REQUIRED_RESULTS["imbalance"]).write_text(json.dumps({
                "selection": {
                    "deployed_configuration": "unweighted",
                    "deployment_gate": {"passed": False},
                }
            }), encoding="utf-8")
            (root / REQUIRED_RESULTS["fitb"]).write_text(json.dumps({
                "coverage": {"retained_questions": 100, "retained_fraction": 0.5},
                "summary": {"lightweight_pairwise_proposed": {
                    "accuracy": {"mean": 0.25}
                }},
            }), encoding="utf-8")
            (root / REQUIRED_RESULTS["domain_shift"]).write_text(json.dumps({
                "overall": {"domain_classifier_cross_validated_auc": 1.0}
            }), encoding="utf-8")
            checkpoint = root / "models/disjoint/compatibility_ranker.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"extra": {
                "checkpoint_version": 2,
                "calibration": {
                    "fit_scope": "validation_only",
                    "validation_nll_before": 0.5,
                    "validation_nll_after": 0.4,
                },
            }}, checkpoint)
            report = run_checks(root)
        self.assertFalse(report["production_promotion_allowed"])
        self.assertIn("official_fitb_above_chance", report["blocking_failures"])


if __name__ == "__main__":
    unittest.main()
