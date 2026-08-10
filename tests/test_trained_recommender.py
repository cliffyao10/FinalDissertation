import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    save_checkpoint,
)
from src.trained_recommender import recommend_with_trained_model


class TrainedRecommendationAdapterTests(unittest.TestCase):
    def test_checkpoint_and_catalogue_drive_recommendation(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "ranker.pt"
            catalogue = root / "catalogue.pt"
            model = CompatibilityRanker(
                ModelConfig(embedding_dim=8, hidden_dim=16, slot_dim=4, dropout=0.0)
            )
            save_checkpoint(checkpoint, model, {"best_validation_metrics": {"auc": 0.75}})
            items = []
            colours = ("Black", "White", "Blue")
            for slot in ("inner_top", "outer_top", "bottom", "shoes"):
                for index, colour in enumerate(colours):
                    items.append(
                        {
                            "item_id": f"{slot}-{index}",
                            "slot": slot,
                            "type": f"Test {slot}",
                            "colour": colour,
                            "style": "Casual",
                            "image_path": "",
                            "embedding": torch.randn(8),
                        }
                    )
            torch.save({"items": items, "model_name": "test-siglip"}, catalogue)

            result = recommend_with_trained_model(
                input_slot="inner_top",
                input_category="T-Shirt",
                input_colour="Blue",
                input_embedding=torch.randn(8),
                style="Casual",
                weather=None,
                slot_labels={
                    "inner_top": "Inner Top",
                    "outer_top": "Outer Layer",
                    "bottom": "Bottom",
                    "shoes": "Shoes",
                },
                checkpoint=checkpoint,
                catalogue=catalogue,
            )

            self.assertEqual(result["method"], "trained_siglip_compatibility_ranker_v1")
            self.assertEqual(len(result["primary"]["items"]), 3)
            self.assertEqual(len(result["alternative"]["items"]), 3)
            self.assertEqual(result["model"]["embedding_model"], "test-siglip")


if __name__ == "__main__":
    unittest.main()
