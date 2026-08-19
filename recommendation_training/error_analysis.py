"""Slice held-out compatibility errors by outfit structure and replaced slot."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset, SLOTS
from recommendation_training.evaluate import classification_metrics, predict
from recommendation_training.fashion_model_comparison import (
    infer_matched_replacement_groups,
)


def expected_calibration_error(labels, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    records = []
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        if count:
            confidence = float(probabilities[selected].mean())
            observed = float(labels[selected].mean())
            error += count / len(labels) * abs(confidence - observed)
        else:
            confidence = observed = None
        records.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "examples": count,
            "mean_probability": confidence,
            "positive_rate": observed,
        })
    return {"bins": bins, "ece": float(error), "reliability": records}


def safe_metrics(labels, probabilities):
    if not len(labels):
        return None
    return classification_metrics(np.asarray(labels), np.asarray(probabilities))


def analyse(dataset, labels, probabilities):
    groups = infer_matched_replacement_groups(dataset)
    by_present_slots = {}
    present_counts = dataset.masks.sum(dim=1).numpy()
    for count in sorted(set(present_counts.tolist())):
        selected = present_counts == count
        by_present_slots[str(count)] = safe_metrics(
            labels[selected], probabilities[selected]
        )

    by_replaced_slot = {}
    for slot_index, slot_name in enumerate(SLOTS):
        selected_groups = [group for group in groups if group["slot"] == slot_index]
        indices = np.asarray(
            [
                index
                for group in selected_groups
                for index in (group["positive_index"], group["negative_index"])
            ],
            dtype=np.int64,
        )
        pair_wins = [
            probabilities[group["positive_index"]]
            > probabilities[group["negative_index"]]
            for group in selected_groups
        ]
        by_replaced_slot[slot_name] = {
            "matched_pairs": len(selected_groups),
            "pair_ranking_accuracy": (
                float(np.mean(pair_wins)) if pair_wins else None
            ),
            "classification": (
                safe_metrics(labels[indices], probabilities[indices])
                if len(indices)
                else None
            ),
        }
    mistakes = np.where((probabilities >= 0.5).astype(int) != labels)[0]
    highest_confidence_mistakes = sorted(
        mistakes.tolist(),
        key=lambda index: abs(float(probabilities[index]) - 0.5),
        reverse=True,
    )[:25]
    return {
        "overall": safe_metrics(labels, probabilities),
        "calibration": expected_calibration_error(labels, probabilities),
        "by_present_slot_count": by_present_slots,
        "by_replaced_slot": by_replaced_slot,
        "highest_confidence_mistakes": [
            {
                "example_index": int(index),
                "label": int(labels[index]),
                "probability": float(probabilities[index]),
                "present_slots": [
                    SLOTS[slot]
                    for slot in range(len(SLOTS))
                    if bool(dataset.masks[index, slot])
                ],
            }
            for index in highest_confidence_mistakes
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="results/compatibility_error_analysis.json")
    args = parser.parse_args()

    dataset = OutfitDataset(args.test)
    model, checkpoint_metadata = load_checkpoint(args.checkpoint)
    labels, probabilities = predict(
        model, DataLoader(dataset, batch_size=args.batch_size), "cpu"
    )
    report = {
        "checkpoint": args.checkpoint,
        "test_dataset": args.test,
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_metadata": dataset.metadata,
        "analysis": analyse(dataset, labels, probabilities),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["analysis"], indent=2))
    print(f"Saved error analysis to {output_path}")


if __name__ == "__main__":
    main()
