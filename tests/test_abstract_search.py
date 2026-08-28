import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.build_abstract_prototypes import select_medoid
from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    save_checkpoint,
)
from recommendation_training.ranker import OutfitCandidateRanker


class AbstractSearchTests(unittest.TestCase):
    def test_preprojected_scoring_matches_normal_forward(self):
        torch.manual_seed(3)
        model = CompatibilityRanker(
            ModelConfig(embedding_dim=8, hidden_dim=12, slot_dim=4, dropout=0.0)
        ).eval()
        embeddings = torch.randn(7, 4, 8)
        mask = torch.ones(7, 4, dtype=torch.bool)
        slot_ids = torch.arange(4).unsqueeze(0).expand(7, -1)
        features = torch.cat(
            (model.image_projection(embeddings), model.slot_embedding(slot_ids)), dim=-1
        )

        self.assertTrue(
            torch.allclose(
                model(embeddings, mask),
                model.score_preprojected(features, mask),
                atol=1e-6,
            )
        )

    def test_medoid_is_an_observed_embedding_near_the_centroid(self):
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=torch.float32
        )
        self.assertEqual(select_medoid(embeddings), 1)

    def test_exact_search_scores_the_complete_cartesian_product(self):
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model = CompatibilityRanker(
                ModelConfig(embedding_dim=8, hidden_dim=12, slot_dim=4, dropout=0.0)
            )
            save_checkpoint(checkpoint, model)
            ranker = OutfitCandidateRanker(checkpoint, device="cpu")
            slots = ("inner_top", "outer_top", "bottom", "shoes")

            def item(slot, index):
                return {
                    "item_id": f"{slot}-{index}",
                    "slot": slot,
                    "type": slot,
                    "colour": str(index),
                    "embedding": torch.randn(8),
                }

            fixed = {"inner_top": item("inner_top", 0)}
            candidates = {
                slot: [item(slot, index) for index in range(3)]
                for slot in slots[1:]
            }
            ranked = ranker.rank_exact(
                fixed, candidates, top_k=27, batch_size=5
            )

            self.assertEqual(len(ranked), 27)
            self.assertEqual(ranker.last_search_stats["combinations_evaluated"], 27)
            self.assertTrue(ranker.last_search_stats["exact"])
            self.assertEqual(
                [row["compatibility_score"] for row in ranked],
                sorted(
                    [row["compatibility_score"] for row in ranked], reverse=True
                ),
            )

    def test_safety_gate_never_returns_a_prefix_as_exact(self):
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(
                checkpoint,
                CompatibilityRanker(
                    ModelConfig(
                        embedding_dim=4, hidden_dim=8, slot_dim=2, dropout=0.0
                    )
                ),
            )
            ranker = OutfitCandidateRanker(checkpoint, device="cpu")
            fixed = {
                "inner_top": {
                    "slot": "inner_top",
                    "embedding": torch.randn(4),
                }
            }
            candidates = {
                slot: [
                    {"slot": slot, "embedding": torch.randn(4)} for _ in range(2)
                ]
                for slot in ("outer_top", "bottom", "shoes")
            }
            with self.assertRaises(RuntimeError):
                ranker.rank_exact(
                    fixed,
                    candidates,
                    top_k=2,
                    batch_size=3,
                    maximum_combinations=7,
                )


if __name__ == "__main__":
    unittest.main()
