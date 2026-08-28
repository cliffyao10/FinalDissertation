import json
import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.evaluate_official_fitb import (
    fitb_metrics,
    prepare_official_fitb,
)


class OfficialFitbTests(unittest.TestCase):
    def test_preparation_rejects_answers_mapped_to_different_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [{
                "set_id": "10",
                "items": [
                    {"index": 1, "categoryid": 17, "image": "x?tid=101"},
                    {"index": 2, "categoryid": 9, "image": "x?tid=102"},
                    {"index": 3, "categoryid": 41, "image": "x?tid=103"},
                    {"index": 4, "categoryid": 23, "image": "x?tid=104"},
                    {"index": 5, "categoryid": 17, "image": "x?tid=105"},
                    {"index": 6, "categoryid": 23, "image": "x?tid=106"},
                    {"index": 7, "categoryid": 23, "image": "x?tid=107"},
                ],
            }]
            questions = [{
                "question": ["10_1", "10_2", "10_3"],
                "answers": ["10_4", "10_5", "10_6", "10_7"],
                "blank_position": 2,
            }]
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
            torch.save({
                "item_ids": [str(value) for value in range(101, 108)],
                "embeddings": torch.randn(7, 8),
                "model_name": "test",
            }, root / "cache.pt")
            payload = prepare_official_fitb(
                root / "metadata.json", root / "questions.json", root / "cache.pt"
            )
            self.assertEqual(payload["retained_questions"], 0)
            self.assertEqual(payload["skipped"]["answer_slot_mismatch"], 1)

    def test_split_item_ids_use_external_category_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [{
                "set_id": "10",
                "items": [
                    {"index": index, "item_id": str(100 + index)}
                    for index in range(1, 8)
                ],
            }]
            categories = {
                "101": {"category_id": "17"},
                "102": {"category_id": "9"},
                "103": {"category_id": "41"},
                **{
                    str(value): {"category_id": "23"}
                    for value in range(104, 108)
                },
            }
            questions = [{
                "question": ["10_1", "10_2", "10_3"],
                "answers": ["10_4", "10_5", "10_6", "10_7"],
                "blank_position": 2,
            }]
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
            (root / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
            torch.save({
                "item_ids": [str(value) for value in range(101, 108)],
                "embeddings": torch.randn(7, 8),
                "model_name": "test",
            }, root / "cache.pt")
            payload = prepare_official_fitb(
                root / "metadata.json",
                root / "questions.json",
                root / "cache.pt",
                item_metadata_path=root / "categories.json",
            )
            self.assertEqual(payload["retained_questions"], 1)

    def test_metrics_rank_first_answer_as_correct(self):
        metrics = fitb_metrics([4, 3, 2, 1, 1, 2, 3, 4])
        self.assertEqual(metrics["questions"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 0.625)

    def test_metrics_support_shuffled_disjoint_answers(self):
        metrics = fitb_metrics(
            [1, 2, 3, 4, 4, 3, 2, 1],
            correct_answer_indices=[3, 0],
        )
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_preparation_derives_shuffled_answer_index_from_set_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [
                {
                    "set_id": set_id,
                    "items": [
                        {
                            "index": index,
                            "categoryid": category,
                            "image": f"x?tid={set_id}{index}",
                        }
                        for index, category in ((1, 17), (2, 9), (3, 41), (4, 23))
                    ],
                }
                for set_id in ("10", "20", "30", "40")
            ]
            questions = [{
                "question": ["10_1", "10_2", "10_3"],
                "answers": ["20_4", "30_4", "10_4", "40_4"],
                "blank_position": 4,
            }]
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
            item_ids = [f"{set_id}{index}" for set_id in ("10", "20", "30", "40") for index in range(1, 5)]
            torch.save({
                "item_ids": item_ids,
                "embeddings": torch.randn(len(item_ids), 8),
                "model_name": "test",
            }, root / "cache.pt")
            payload = prepare_official_fitb(
                root / "metadata.json", root / "questions.json", root / "cache.pt"
            )
            self.assertEqual(payload["correct_answer_indices"], [2])

    def test_preparation_maps_official_references_to_four_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [{
                "set_id": "10",
                "items": [
                    {"index": 1, "categoryid": 17, "image": "x?tid=101"},
                    {"index": 2, "categoryid": 9, "image": "x?tid=102"},
                    {"index": 3, "categoryid": 41, "image": "x?tid=103"},
                    {"index": 4, "categoryid": 23, "image": "x?tid=104"},
                    {"index": 5, "categoryid": 23, "image": "x?tid=105"},
                    {"index": 6, "categoryid": 23, "image": "x?tid=106"},
                    {"index": 7, "categoryid": 23, "image": "x?tid=107"},
                ],
            }]
            questions = [{
                "question": ["10_1", "10_2", "10_3"],
                "answers": ["10_4", "10_5", "10_6", "10_7"],
                "blank_position": 2,
            }]
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
            torch.save({
                "item_ids": [str(value) for value in range(101, 108)],
                "embeddings": torch.randn(7, 8),
                "model_name": "test",
            }, root / "cache.pt")
            payload = prepare_official_fitb(
                root / "metadata.json", root / "questions.json", root / "cache.pt"
            )
            self.assertEqual(payload["retained_questions"], 1)
            self.assertEqual(tuple(payload["embeddings"].shape), (4, 4, 8))


if __name__ == "__main__":
    unittest.main()
