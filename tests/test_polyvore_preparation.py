import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from recommendation_training.prepare_polyvore import (
    build_examples,
    create_embedding_cache,
    image_identifier,
    load_split,
)


class PolyvorePreparationTests(unittest.TestCase):
    def test_reads_item_id_from_polyvore_image_url(self):
        self.assertEqual(
            image_identifier(
                {
                    "image": (
                        "http://img2.polyvoreimg.com/cgi/img-thing?"
                        ".out=jpg&size=m&tid=194508109"
                    )
                }
            ),
            "194508109",
        )

    def test_load_split_supports_item_id_image_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_directory = root / "images"
            image_directory.mkdir()
            raw_items = []
            for index, (category, item_id) in enumerate(
                ((21, "101"), (25, "102"), (28, "103")),
                start=1,
            ):
                (image_directory / f"{item_id}.jpg").touch()
                raw_items.append(
                    {
                        "index": index,
                        "categoryid": category,
                        "image": f"https://example.test/image?tid={item_id}",
                    }
                )
            metadata = root / "split.json"
            metadata.write_text(
                json.dumps([{"set_id": "outfit", "items": raw_items}]),
                encoding="utf-8",
            )

            outfits, items = load_split(metadata, root)

            self.assertEqual(len(outfits), 1)
            self.assertEqual(set(items), {"101", "102", "103"})

    def test_embedding_cache_resumes_only_missing_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.pt"
            items = {}
            for item_id in ("one", "two", "three"):
                path = root / f"{item_id}.jpg"
                path.touch()
                items[item_id] = {"path": str(path)}
            torch.save(
                {
                    "item_ids": ["one"],
                    "embeddings": torch.ones(1, 4),
                    "model_name": "test-model",
                },
                cache_path,
            )

            with mock.patch(
                "recommendation_training.prepare_polyvore.load_encoder",
                return_value=(object(), object(), "cpu"),
            ) as load_encoder, mock.patch(
                "recommendation_training.prepare_polyvore.embed_paths",
                return_value=torch.full((2, 4), 2.0),
            ) as embed_paths:
                payload = create_embedding_cache(
                    items,
                    cache_path,
                    "test-model",
                    batch_size=2,
                    save_every_batches=1,
                )

            load_encoder.assert_called_once()
            self.assertEqual(len(embed_paths.call_args.args[0]), 2)
            self.assertEqual(payload["item_ids"], ["one", "three", "two"])
            self.assertEqual(tuple(payload["embeddings"].shape), (3, 4))

            with mock.patch(
                "recommendation_training.prepare_polyvore.load_encoder"
            ) as load_encoder:
                resumed = create_embedding_cache(
                    items,
                    cache_path,
                    "test-model",
                    batch_size=2,
                )
            load_encoder.assert_not_called()
            self.assertEqual(resumed["item_ids"], payload["item_ids"])

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
