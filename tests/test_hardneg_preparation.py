import json
import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.prepare_hardneg import (
    build_hard_negative_examples,
    build_item_lookup,
    parse_compatibility_rows,
)


class HardNegativePreparationTests(unittest.TestCase):
    def test_maps_published_keys_and_builds_masked_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "test.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "set_id": "100",
                            "items": [
                                {
                                    "index": index,
                                    "categoryid": category,
                                    "image": f"https://example.test/?tid={item_id}",
                                }
                                for index, category, item_id in (
                                    (1, 21, "11"),
                                    (2, 25, "12"),
                                    (3, 28, "13"),
                                    (4, 49, "14"),
                                )
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            compatibility = root / "compatibility.txt"
            compatibility.write_text(
                "1 100_1 100_2 100_3 100_4\n0 100_1 100_2 100_3\n",
                encoding="utf-8",
            )
            lookup = build_item_lookup(metadata)
            embedding_by_id = {
                item_id: torch.full((8,), float(item_id))
                for item_id in ("11", "12", "13", "14")
            }

            payload = build_hard_negative_examples(
                parse_compatibility_rows(compatibility),
                lookup,
                embedding_by_id,
            )

            self.assertEqual(set(lookup), {"100_1", "100_2", "100_3", "100_4"})
            self.assertEqual(tuple(payload["embeddings"].shape), (2, 4, 8))
            self.assertEqual(payload["positive_examples"], 1)
            self.assertEqual(payload["negative_examples"], 1)
            self.assertEqual(payload["masks"].sum(dim=1).tolist(), [4, 3])

    def test_skips_rows_without_three_cached_supported_slots(self):
        rows = [(0.0, ["100_1", "100_2", "100_3"])]
        lookup = {
            "100_1": {"item_id": "11", "slot": "inner_top"},
            "100_2": {"item_id": "12", "slot": "bottom"},
            "100_3": {"item_id": "13", "slot": "shoes"},
        }
        with self.assertRaises(RuntimeError):
            build_hard_negative_examples(
                rows,
                lookup,
                {"11": torch.ones(8), "12": torch.ones(8)},
            )


if __name__ == "__main__":
    unittest.main()
