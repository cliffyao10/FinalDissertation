import unittest

import torch
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.train import evaluate


class _UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.embeddings = torch.randn(3, 4, 8)
        self.masks = torch.ones(3, 4, dtype=torch.bool)
        self.labels = torch.tensor([1.0, 0.0, 1.0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "index": index,
            "embeddings": self.embeddings[index],
            "mask": self.masks[index],
            "label": self.labels[index],
        }


class TrainEvaluationTests(unittest.TestCase):
    def test_published_unpaired_rows_keep_overall_metrics(self):
        dataset = _UnpairedDataset()
        model = CompatibilityRanker(
            ModelConfig(embedding_dim=8, hidden_dim=16, slot_dim=4, dropout=0.0)
        )

        metrics = evaluate(
            model,
            DataLoader(dataset, batch_size=3),
            "cpu",
            dataset,
        )

        self.assertIn("auc", metrics)
        self.assertIn("groups_unavailable_reason", metrics)
        self.assertNotIn("groups", metrics)


if __name__ == "__main__":
    unittest.main()
