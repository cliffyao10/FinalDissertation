import unittest

from recommendation_training.overfitting_audit import interpret


class OverfittingAuditTest(unittest.TestCase):
    def test_visible_training_gap_can_still_have_stable_holdout(self):
        report = interpret(
            0.97,
            0.86,
            0.858,
            [
                {"epoch": 1, "validation_auc": 0.80},
                {"epoch": 2, "validation_auc": 0.86},
                {"epoch": 3, "validation_auc": 0.85},
            ],
            2,
        )
        self.assertTrue(report["training_fit_materially_higher"])
        self.assertTrue(report["validation_test_agreement_within_0_02"])
        self.assertEqual(report["decision"], "retain_frozen_checkpoint")


if __name__ == "__main__":
    unittest.main()
