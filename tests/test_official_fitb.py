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
    def test_metrics_rank_first_answer_as_correct(self):
        metrics = fitb_metrics([4, 3, 2, 1, 1, 2, 3, 4])
        self.assertEqual(metrics["questions"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 0.625)

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
