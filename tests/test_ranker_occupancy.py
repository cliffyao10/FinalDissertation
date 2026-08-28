import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    save_checkpoint,
)
from recommendation_training.ranker import OutfitCandidateRanker, item_occupied_slots


class RankerOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        checkpoint = Path(self.directory.name) / "ranker.pt"
        model = CompatibilityRanker(
            ModelConfig(embedding_dim=4, hidden_dim=8, slot_dim=2, dropout=0.0)
        )
        save_checkpoint(checkpoint, model, {})
        self.ranker = OutfitCandidateRanker(checkpoint, device="cpu")

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def item(item_id, slot, occupied=None):
        return {
            "item_id": item_id,
            "slot": slot,
            "colour": "Black",
            "embedding": torch.randn(4),
            "occupies_slots": occupied or slot,
        }

    def test_parses_serialised_multi_slot_value(self):
        self.assertEqual(
            item_occupied_slots(
                {"slot": "inner_top", "occupies_slots": "inner_top;bottom"}
            ),
            {"inner_top", "bottom"},
        )

    def test_fixed_dress_suppresses_bottom_candidate(self):
        ranked = self.ranker.rank(
            {
                "inner_top": self.item(
                    "dress", "inner_top", "inner_top;bottom"
                )
            },
            {
                "outer_top": [self.item("coat", "outer_top")],
                "shoes": [self.item("shoes", "shoes")],
            },
        )

        self.assertEqual(
            {item["item_id"] for item in ranked[0]["items"]},
            {"dress", "coat", "shoes"},
        )
        self.assertNotIn(
            "bottom", {item["slot"] for item in ranked[0]["items"]}
        )

    def test_candidate_dress_suppresses_bottom_candidate(self):
        ranked = self.ranker.rank(
            {"shoes": self.item("shoes", "shoes")},
            {
                "inner_top": [
                    self.item("dress", "inner_top", "inner_top;bottom")
                ],
                "outer_top": [self.item("coat", "outer_top")],
                "bottom": [self.item("trousers", "bottom")],
            },
        )

        self.assertEqual(
            {item["item_id"] for item in ranked[0]["items"]},
            {"dress", "coat", "shoes"},
        )

    def test_candidate_dress_cannot_replace_fixed_bottom(self):
        ranked = self.ranker.rank(
            {"bottom": self.item("uploaded-trousers", "bottom")},
            {
                "inner_top": [
                    self.item("dress", "inner_top", "inner_top;bottom"),
                    self.item("shirt", "inner_top"),
                ],
                "outer_top": [self.item("coat", "outer_top")],
                "shoes": [self.item("shoes", "shoes")],
            },
        )

        self.assertEqual(
            {item["item_id"] for item in ranked[0]["items"]},
            {"shirt", "coat", "uploaded-trousers", "shoes"},
        )


if __name__ == "__main__":
    unittest.main()
