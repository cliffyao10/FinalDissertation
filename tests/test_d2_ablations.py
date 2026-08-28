import unittest

import numpy as np

from recommendation_training.evaluate_d2_ablations import (
    correct_vector,
    paired_question_bootstrap,
)


class D2AblationTest(unittest.TestCase):
    def test_correct_vector_respects_shuffled_targets(self):
        scores = [0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.1, 0.0]
        np.testing.assert_array_equal(
            correct_vector(scores, [2, 1]), np.asarray([True, False])
        )

    def test_paired_bootstrap_detects_uniform_improvement(self):
        full = [np.ones(20, dtype=bool) for _ in range(3)]
        ablated = [np.zeros(20, dtype=bool) for _ in range(3)]
        report = paired_question_bootstrap(
            full, ablated, seed=4, iterations=100
        )
        self.assertEqual(
            report["mean_accuracy_difference_full_minus_ablation"], 1.0
        )
        self.assertEqual(
            report["paired_question_bootstrap_95_percent_ci"], [1.0, 1.0]
        )


if __name__ == "__main__":
    unittest.main()
