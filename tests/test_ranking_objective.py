import unittest

import torch

from recommendation_training.ranking_objective_study import hybrid_loss


class RankingObjectiveTests(unittest.TestCase):
    def test_ranking_penalty_is_zero_above_margin(self):
        total, classification, ranking = hybrid_loss(
            torch.tensor([1.0]), torch.tensor([0.0]), margin=0.2, ranking_weight=0.5
        )
        self.assertEqual(ranking.item(), 0.0)
        self.assertAlmostEqual(total.item(), classification.item())

    def test_ranking_penalty_activates_for_reversed_pair(self):
        _, _, ranking = hybrid_loss(
            torch.tensor([0.0]), torch.tensor([1.0]), margin=0.2, ranking_weight=0.5
        )
        self.assertAlmostEqual(ranking.item(), 1.2, places=5)


if __name__ == "__main__":
    unittest.main()
