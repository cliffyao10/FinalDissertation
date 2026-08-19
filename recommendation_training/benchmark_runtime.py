"""Measure lightweight model and end-to-end catalogue-ranking latency."""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset, SLOTS
from src.trained_recommender import recommend_with_trained_model


SLOT_LABELS = {
    "inner_top": "Inner top",
    "outer_top": "Outer layer",
    "bottom": "Bottom",
    "shoes": "Shoes",
}


def latency_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "iterations": int(len(values)),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
    }


def timed_calls(function, iterations):
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        durations.append((time.perf_counter() - started) * 1000)
    return latency_summary(durations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--catalogue", default="models/catalogue_embeddings.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--output", default="results/model_runtime_benchmark.json")
    parser.add_argument("--head-iterations", type=int, default=300)
    parser.add_argument("--ranking-iterations", type=int, default=10)
    args = parser.parse_args()

    device = "cpu"
    model, metadata = load_checkpoint(args.checkpoint, device)
    dataset = OutfitDataset(args.test)
    embeddings = dataset.embeddings[:1]
    masks = dataset.masks[:1]
    with torch.inference_mode():
        for _ in range(20):
            model(embeddings, masks)
        head_latency = timed_calls(
            lambda: model(embeddings, masks), args.head_iterations
        )

    present_slot_index = int(torch.where(masks[0])[0][0])
    input_slot = SLOTS[present_slot_index]
    input_embedding = embeddings[0, present_slot_index]

    def rank_catalogue():
        return recommend_with_trained_model(
            input_slot=input_slot,
            input_category="Garment",
            input_colour="Neutral",
            input_embedding=input_embedding,
            style="Casual",
            weather=None,
            slot_labels=SLOT_LABELS,
            checkpoint=Path(args.checkpoint),
            catalogue=Path(args.catalogue),
        )

    rank_catalogue()  # Exclude one-time checkpoint/catalogue loading.
    ranking_latency = timed_calls(rank_catalogue, args.ranking_iterations)
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    report = {
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size,
        "catalogue_embedding_size_bytes": Path(args.catalogue).stat().st_size,
        "single_outfit_head_latency": head_latency,
        "warm_end_to_end_catalogue_ranking_latency": ranking_latency,
        "notes": {
            "warm_runtime": "One-time artifact loading is excluded from ranking latency.",
            "input": f"One held-out {input_slot} embedding with a Casual style filter.",
        },
        "checkpoint_metadata": metadata,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved runtime benchmark to {output_path}")


if __name__ == "__main__":
    main()
