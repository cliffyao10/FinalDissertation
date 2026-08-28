import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import (
    bootstrap_auc_interval,
    classification_metrics,
    predict,
)


class TrainingEvaluationTests(unittest.TestCase):
    def test_predict_supports_uncalibrated_baseline_logits(self):
        class Baseline(torch.nn.Module):
            def forward(self, embeddings, mask):
                return embeddings[:, 0, 0]

        payload = {
            "embeddings": torch.tensor([[[0.0]], [[2.0]]]),
            "mask": torch.ones(2, 1, dtype=torch.bool),
            "label": torch.tensor([0.0, 1.0]),
        }
        labels, probabilities = predict(Baseline(), [payload], "cpu")
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertGreater(probabilities[1], 0.5)

    def test_dataset_exposes_reproducibility_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.pt"
            torch.save(
                {
                    "embeddings": torch.randn(4, 4, 8),
                    "masks": torch.ones(4, 4, dtype=torch.bool),
                    "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                    "siglip_model": "test-encoder",
                    "seed": 42,
                },
                path,
            )

            dataset = OutfitDataset(path)

            self.assertEqual(dataset.metadata["siglip_model"], "test-encoder")
            self.assertEqual(dataset.metadata["seed"], 42)
            self.assertEqual(dataset.positive_rate, 0.5)

    def test_metrics_and_bootstrap_interval_are_deterministic(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.3, 0.7, 0.9])

        metrics = classification_metrics(labels, probabilities)
        first = bootstrap_auc_interval(labels, probabilities, iterations=100, seed=7)
        second = bootstrap_auc_interval(labels, probabilities, iterations=100, seed=7)

        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["matthews_correlation_coefficient"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(first, second)
        self.assertEqual(first, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
