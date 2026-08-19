import unittest

import torch

from recommendation_training.ablation_study import AblationRanker, masked_mean
from recommendation_training.compatibility_model import ModelConfig


class AblationStudyTests(unittest.TestCase):
    def test_masked_mean_ignores_missing_slots(self):
        embeddings = torch.tensor([[[1.0, 3.0], [9.0, 9.0], [3.0, 5.0], [7.0, 7.0]]])
        masks = torch.tensor([[True, False, True, False]])
        self.assertTrue(torch.equal(masked_mean(embeddings, masks), torch.tensor([[2.0, 4.0]])))

    def test_every_neural_ablation_returns_one_logit_per_outfit(self):
        config = ModelConfig(embedding_dim=8, hidden_dim=6, slot_dim=2)
        embeddings = torch.randn(3, 4, 8)
        masks = torch.tensor([
            [True, True, True, True],
            [True, False, True, True],
            [False, True, True, True],
        ])
        for use_slot in (False, True):
            for use_pairwise in (False, True):
                with self.subTest(slot=use_slot, pairwise=use_pairwise):
                    model = AblationRanker(
                        config,
                        use_slot_embedding=use_slot,
                        use_pairwise=use_pairwise,
                    )
                    self.assertEqual(tuple(model(embeddings, masks).shape), (3,))


if __name__ == "__main__":
    unittest.main()
