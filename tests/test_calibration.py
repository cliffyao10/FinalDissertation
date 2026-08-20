import unittest

import torch

from recommendation_training.calibrate_checkpoint import fit_temperature
from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig


class CalibrationTests(unittest.TestCase):
    def test_temperature_changes_scale_not_order(self):
        model = CompatibilityRanker(ModelConfig(embedding_dim=2, hidden_dim=8, slot_dim=2))
        model.eval()
        model.forward = lambda embeddings, mask: torch.tensor(
            [-2.0, -0.5, 0.5, 2.0]
        )
        embeddings = torch.zeros(4, 4, 2)
        mask = torch.ones(4, 4, dtype=torch.bool)
        before = model.probability(embeddings, mask)
        model.calibration_temperature = 2.0
        after = model.probability(embeddings, mask)
        self.assertTrue(torch.equal(torch.argsort(before), torch.argsort(after)))

    def test_overconfident_logits_receive_temperature_above_one(self):
        logits = torch.tensor([8.0, -8.0, 8.0, -8.0])
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
        self.assertGreater(fit_temperature(logits, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
