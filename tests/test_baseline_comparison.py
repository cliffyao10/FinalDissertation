import unittest

import numpy as np
import torch

from recommendation_training.compare_baselines import mean_pairwise_cosine


class BaselineComparisonTests(unittest.TestCase):
    def test_pairwise_cosine_respects_mask(self):
        embeddings = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        masks = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )

        scores = mean_pairwise_cosine(embeddings, masks)

        np.testing.assert_allclose(scores, [1.0, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
