import unittest

import numpy as np
import torch

from recommendation_training.disjoint_imbalance_study import (
    length_diagnostics,
    length_weights,
)


class Dataset:
    masks = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
    )

    def __len__(self):
        return len(self.masks)


class DisjointImbalanceTests(unittest.TestCase):
    def test_length_weights_upweight_rare_length(self):
        weights = length_weights(Dataset(), exponent=0.5, maximum=3.0)
        self.assertGreater(weights[2], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_length_diagnostics_do_not_require_adjacent_pairs(self):
        labels = np.asarray([1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.2, 0.8, 0.1])
        dataset = Dataset()
        dataset.masks = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.bool,
        )
        report = length_diagnostics(dataset, labels, probabilities)
        self.assertEqual(set(report["by_present_slot_count"]), {"3", "4"})
        self.assertEqual(report["worst_length_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
