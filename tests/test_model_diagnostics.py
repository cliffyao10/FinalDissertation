import unittest

import numpy as np

from recommendation_training.domain_shift import domain_metrics
from recommendation_training.error_analysis import expected_calibration_error


class ModelDiagnosticTests(unittest.TestCase):
    def test_calibration_error_is_zero_for_matching_bins(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        result = expected_calibration_error(labels, probabilities, bins=2)
        self.assertEqual(result["ece"], 0.0)

    def test_domain_metrics_detect_identical_samples(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(20, 8))
        result = domain_metrics(values, values.copy(), seed=7)
        self.assertAlmostEqual(result["centroid_cosine_similarity"], 1.0)
        self.assertAlmostEqual(result["catalogue_to_polyvore_nearest_cosine_mean"], 1.0)
        self.assertLess(abs(result["rbf_mmd_squared"]), 0.1)


if __name__ == "__main__":
    unittest.main()
