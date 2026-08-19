"""Create frozen SigLIP vectors for the product candidate catalogue."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from recommendation_training.dataset import SLOTS
from recommendation_training.embedding import DEFAULT_MODEL, embed_paths, load_encoder


REQUIRED_COLUMNS = {"item_id", "slot", "type", "colour", "image_path"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", default="data/catalogue.csv")
    parser.add_argument("--output", default="models/catalogue_embeddings.pt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame = pd.read_csv(args.catalogue)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Catalogue is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Catalogue has no product rows.")
    invalid_slots = sorted(set(frame["slot"]) - set(SLOTS))
    if invalid_slots:
        raise ValueError(f"Unsupported slots: {invalid_slots}")
    paths = [Path(value) for value in frame["image_path"]]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing catalogue images, first: {missing_paths[0]}")

    processor, model, device = load_encoder(args.model)
    batches = []
    for start in range(0, len(paths), args.batch_size):
        batches.append(
            embed_paths(paths[start : start + args.batch_size], processor, model, device)
        )
    records = frame.fillna("").to_dict(orient="records")
    for record, embedding in zip(records, torch.cat(batches)):
        record["embedding"] = embedding
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"items": records, "model_name": args.model}, output_path)
    print(f"Saved {len(records)} catalogue items to {output_path}")


if __name__ == "__main__":
    main()
