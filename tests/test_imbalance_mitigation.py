import unittest

from recommendation_training.imbalance_mitigation_study import (
    choose_configuration,
    deployment_gate,
    selection_score,
)


class ImbalanceMitigationTests(unittest.TestCase):
    def test_selection_score_includes_weakest_group(self):
        groups = {
            "macro_slot_auc": 0.8,
            "worst_slot_auc": 0.5,
            "macro_length_auc": 0.7,
        }
        self.assertAlmostEqual(selection_score(0.8, groups), 0.7)

    def test_configuration_selection_uses_validation_only(self):
        runs = {
            "unweighted": [{
                "best_validation_balanced_selection_score": 0.7,
                "checkpoint": "baseline.pt",
                "test_metrics": {"auc": 0.99},
            }],
            "balanced": [{
                "best_validation_balanced_selection_score": 0.8,
                "checkpoint": "balanced.pt",
                "test_metrics": {"auc": 0.1},
            }],
        }
        chosen, _, checkpoint = choose_configuration(runs)
        self.assertEqual(chosen, "balanced")
        self.assertEqual(checkpoint, "balanced.pt")

    def test_deployment_gate_rejects_material_auc_loss(self):
        def metrics(auc, macro, worst, gap):
            return {
                "auc": {"mean": auc},
                "macro_slot_auc": {"mean": macro},
                "worst_slot_auc": {"mean": worst},
                "slot_auc_gap": {"mean": gap},
            }
        summary = {
            "unweighted": metrics(0.74, 0.72, 0.62, 0.14),
            "balanced": metrics(0.70, 0.74, 0.68, 0.08),
        }
        self.assertFalse(deployment_gate(summary, "balanced")["passed"])

    def test_deployment_gate_accepts_negligible_macro_difference(self):
        def metrics(auc, macro, worst, gap):
            return {
                "auc": {"mean": auc},
                "macro_slot_auc": {"mean": macro},
                "worst_slot_auc": {"mean": worst},
                "slot_auc_gap": {"mean": gap},
            }
        summary = {
            "unweighted": metrics(0.740, 0.72730, 0.654, 0.117),
            "balanced": metrics(0.738, 0.72724, 0.664, 0.093),
        }
        self.assertTrue(deployment_gate(summary, "balanced")["passed"])


if __name__ == "__main__":
    unittest.main()
