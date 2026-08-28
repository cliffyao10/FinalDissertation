import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    save_checkpoint,
)
from src.trained_recommender import (
    _audience_filter,
    _shortlist_candidates,
    _style_filter,
    _weather_filter,
    recommend_with_trained_model,
)


class TrainedRecommendationAdapterTests(unittest.TestCase):
    def test_audience_filter_never_mixes_clothing_ranges(self):
        items = [
            {"item_id": "trousers", "audience": "menswear"},
            {"item_id": "skirt", "audience": "womenswear"},
            {"item_id": "neutral", "audience": "neutral"},
        ]
        self.assertEqual(
            {item["item_id"] for item in _audience_filter(items, "menswear")},
            {"trousers", "neutral"},
        )
        self.assertEqual(
            {item["item_id"] for item in _audience_filter(items, "womenswear")},
            {"skirt", "neutral"},
        )
    def test_shortlist_depends_on_input_and_keeps_colour_variety(self):
        items = [
            {
                "item_id": item_id,
                "colour": colour,
                "embedding": torch.tensor(embedding),
            }
            for item_id, colour, embedding in (
                ("red-near", "Red", [1.0, 0.0]),
                ("red-second", "Red", [0.9, 0.1]),
                ("blue", "Blue", [0.7, 0.3]),
                ("green", "Green", [0.5, 0.5]),
                ("black", "Black", [0.0, 1.0]),
            )
        ]

        shortlist = _shortlist_candidates(
            items,
            torch.tensor([1.0, 0.0]),
            limit=4,
            diverse_colours=3,
        )

        self.assertEqual(shortlist[0]["item_id"], "red-near")
        self.assertGreaterEqual(len({item["colour"] for item in shortlist}), 3)

    def test_style_filter_preserves_candidate_coverage_per_slot(self):
        items = [
            {"item_id": "top", "slot": "inner_top", "style": "Casual"},
            {"item_id": "coat", "slot": "outer_top", "style": "Formal"},
            {"item_id": "jeans", "slot": "bottom", "style": "Casual"},
            {"item_id": "shoes", "slot": "shoes", "style": "Formal"},
        ]

        retained = _style_filter(items, "Casual")

        self.assertEqual(
            {item["slot"] for item in retained},
            {"inner_top", "outer_top", "bottom", "shoes"},
        )

    def test_precipitation_filter_requires_protected_outerwear_and_shoes(self):
        items = [
            {
                "item_id": "plain-coat",
                "slot": "outer_top",
                "minimum_temperature": -10,
                "maximum_temperature": 30,
                "weather_tags": "cold",
            },
            {
                "item_id": "rain-coat",
                "slot": "outer_top",
                "minimum_temperature": -10,
                "maximum_temperature": 30,
                "weather_tags": "rain;waterproof",
            },
            {
                "item_id": "plain-shoes",
                "slot": "shoes",
                "minimum_temperature": -10,
                "maximum_temperature": 30,
                "weather_tags": "",
            },
            {
                "item_id": "boots",
                "slot": "shoes",
                "minimum_temperature": -10,
                "maximum_temperature": 30,
                "weather_tags": "water-resistant",
            },
        ]

        retained, constraints = _weather_filter(
            items,
            {"feels_like": 12, "rain_probability": 80, "condition": "Rain"},
        )

        self.assertEqual(
            {item["item_id"] for item in retained},
            {"rain-coat", "boots"},
        )
        self.assertEqual(constraints, ["temperature", "precipitation"])

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
            self.assertEqual(
                {item["type"] for item in result["primary"]["items"]},
                {"Jacket", "Trousers", "Shoes"},
            )
            self.assertEqual(result["model"]["embedding_model"], "test-siglip")
            self.assertEqual(
                set(result["primary"]["alternatives_by_slot"]),
                {"outer_top", "bottom", "shoes"},
            )
            for candidates in result["primary"]["alternatives_by_slot"].values():
                self.assertLessEqual(len(candidates), 3)
                for candidate in candidates:
                    self.assertEqual(
                        candidate["selection_source"],
                        "trained_model_ranked_slot_replacement",
                    )
                    self.assertIn("compatibility_score", candidate)
            current_by_slot = {
                item["slot"]: item for item in result["primary"]["items"]
            }
            for slot, candidates in result["primary"]["alternatives_by_slot"].items():
                for candidate in candidates:
                    self.assertEqual(candidate["type"], current_by_slot[slot]["type"])

    def test_abstract_catalogue_is_searched_exactly_and_never_exposes_images(self):
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "ranker.pt"
            catalogue = root / "abstract.pt"
            save_checkpoint(
                checkpoint,
                CompatibilityRanker(
                    ModelConfig(
                        embedding_dim=8, hidden_dim=16, slot_dim=4, dropout=0.0
                    )
                ),
            )
            items = []
            for slot in ("inner_top", "outer_top", "bottom", "shoes"):
                for index, colour in enumerate(("Black", "Blue", "Red")):
                    items.append(
                        {
                            "item_id": f"abstract-{slot}-{index}",
                            "slot": slot,
                            "type": {
                                "inner_top": "T-Shirt",
                                "outer_top": "Jacket",
                                "bottom": "Trousers",
                                "shoes": "Trainers",
                            }[slot],
                            "colour": colour,
                            "style": "Casual",
                            "image_path": "",
                            "abstract": True,
                            "embedding": torch.randn(8),
                        }
                    )
            torch.save({"items": items, "model_name": "test-siglip"}, catalogue)

            result = recommend_with_trained_model(
                input_slot="inner_top",
                input_category="T-Shirt",
                input_colour="White",
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

            self.assertEqual(
                result["method"], "cove_v3_abstract_exact_search"
            )
            self.assertEqual(result["system_version"], "3.0")
            self.assertTrue(result["model"]["search"]["exact"])
            self.assertEqual(
                result["model"]["search"]["combinations_evaluated"], 27
            )
            for outfit in (result["primary"], result["alternative"]):
                self.assertTrue(all(not item["image_path"] for item in outfit["items"]))

            hot_result = recommend_with_trained_model(
                input_slot="inner_top",
                input_category="T-Shirt",
                input_colour="White",
                input_embedding=torch.randn(8),
                style="Casual",
                weather={"feels_like": 32, "rain_probability": 0},
                slot_labels={
                    "inner_top": "Inner Top",
                    "outer_top": "Outer Layer",
                    "bottom": "Bottom",
                    "shoes": "Shoes",
                },
                checkpoint=checkpoint,
                catalogue=catalogue,
            )
            hot_outer = next(
                item
                for item in hot_result["primary"]["items"]
                if item["slot"] == "outer_top"
            )
            self.assertEqual(hot_outer["type"], "No Outer Layer")
            self.assertIsNone(hot_outer["colour"])


if __name__ == "__main__":
    unittest.main()
