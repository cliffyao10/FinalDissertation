import unittest

import numpy as np
import torch

from recommendation_training.balance import example_weights, group_diagnostics


class TinyDataset:
    def __init__(self):
        self.labels = torch.tensor([1.0, 0.0] * 3)
        self.masks = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0],
             [1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.bool,
        )
        self.embeddings = torch.zeros(6, 4, 2)
        self.embeddings[1, 0] = 1
        self.embeddings[3, 0] = 2
        self.embeddings[5, 1] = 1

    def __len__(self):
        return len(self.labels)


class BalanceTests(unittest.TestCase):
    def test_inverse_frequency_upweights_rare_group_and_preserves_pairs(self):
        weights = example_weights(TinyDataset(), "slot_length", exponent=1.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(float(weights[0]), float(weights[1]))
        self.assertEqual(float(weights[4]), float(weights[5]))
        self.assertGreater(float(weights[4]), float(weights[0]))

    def test_group_diagnostics_reports_slot_disparity(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.1, 0.8, 0.2, 0.4, 0.6])
        report = group_diagnostics(TinyDataset(), labels, probabilities)
        self.assertEqual(report["by_replaced_slot"]["inner_top"]["pairs"], 2)
        self.assertEqual(report["by_replaced_slot"]["outer_top"]["pairs"], 1)
        self.assertGreater(report["slot_auc_gap"], 0)


if __name__ == "__main__":
    unittest.main()
