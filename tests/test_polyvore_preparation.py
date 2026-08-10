import json
import tempfile
import unittest
from pathlib import Path

import torch

from recommendation_training.prepare_polyvore import build_examples, load_split


class PolyvorePreparationTests(unittest.TestCase):
    def test_reads_slots_and_builds_same_slot_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_outfits = []
            for set_id in ("100", "200"):
                items = []
                for index, category in enumerate((21, 25, 28, 49), start=1):
                    image = root / set_id / f"{index}.jpg"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.touch()
                    items.append({"index": str(index), "categoryid": category})
                raw_outfits.append({"set_id": set_id, "items": items})
            metadata = root / "train_no_dup.json"
            metadata.write_text(json.dumps(raw_outfits), encoding="utf-8")

            outfits, items = load_split(metadata, root)
            embedding_by_id = {
                item_id: torch.full((8,), float(index))
                for index, item_id in enumerate(items, start=1)
            }
            payload = build_examples(outfits, items, embedding_by_id, seed=42)

            self.assertEqual(len(outfits), 2)
            self.assertEqual(payload["positive_examples"], 2)
            self.assertEqual(payload["negative_examples"], 2)
            self.assertEqual(tuple(payload["embeddings"].shape), (4, 4, 8))
            self.assertTrue(bool(payload["masks"].all()))


if __name__ == "__main__":
    unittest.main()
