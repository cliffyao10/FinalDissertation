"""Measure frozen-feature shift between Polyvore tests and product candidates."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from recommendation_training.dataset import OutfitDataset, SLOTS


def normalise(values):
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def rbf_mmd_squared(left, right):
    combined = np.concatenate((left, right), axis=0)
    squared = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
    nonzero = squared[squared > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    bandwidth = max(bandwidth, 1e-12)
    kernel = np.exp(-squared / (2.0 * bandwidth))
    left_count = len(left)
    k_xx = kernel[:left_count, :left_count]
    k_yy = kernel[left_count:, left_count:]
    k_xy = kernel[:left_count, left_count:]
    if left_count > 1:
        xx = (k_xx.sum() - np.trace(k_xx)) / (left_count * (left_count - 1))
    else:
        xx = 0.0
    if len(right) > 1:
        yy = (k_yy.sum() - np.trace(k_yy)) / (len(right) * (len(right) - 1))
    else:
        yy = 0.0
    return float(xx + yy - 2.0 * k_xy.mean()), bandwidth


def domain_metrics(polyvore, catalogue, seed=42):
    polyvore = normalise(np.asarray(polyvore, dtype=np.float64))
    catalogue = normalise(np.asarray(catalogue, dtype=np.float64))
    centroid_cosine = float(
        np.dot(polyvore.mean(axis=0), catalogue.mean(axis=0))
        / (
            np.linalg.norm(polyvore.mean(axis=0))
            * np.linalg.norm(catalogue.mean(axis=0))
        )
    )
    similarities = catalogue @ polyvore.T
    nearest = similarities.max(axis=1)
    sample_size = min(len(polyvore), len(catalogue))
    rng = np.random.default_rng(seed)
    balanced_polyvore = polyvore[
        rng.choice(len(polyvore), size=sample_size, replace=False)
    ]
    balanced_catalogue = catalogue[
        rng.choice(len(catalogue), size=sample_size, replace=False)
    ]
    features = np.concatenate((balanced_polyvore, balanced_catalogue), axis=0)
    labels = np.concatenate((np.zeros(sample_size), np.ones(sample_size)))
    folds = min(5, sample_size)
    classifier = LogisticRegression(max_iter=2000, random_state=seed)
    predictions = cross_val_predict(
        classifier,
        features,
        labels,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    mmd, bandwidth = rbf_mmd_squared(balanced_polyvore, balanced_catalogue)
    return {
        "polyvore_items": int(len(polyvore)),
        "catalogue_items": int(len(catalogue)),
        "centroid_cosine_similarity": centroid_cosine,
        "catalogue_to_polyvore_nearest_cosine_mean": float(nearest.mean()),
        "catalogue_to_polyvore_nearest_cosine_min": float(nearest.min()),
        "domain_classifier_cross_validated_auc": float(roc_auc_score(labels, predictions)),
        "rbf_mmd_squared": mmd,
        "rbf_bandwidth_median_heuristic": bandwidth,
        "interpretation": (
            "Domain-classifier AUC near 0.5 indicates overlapping feature "
            "distributions; values near 1.0 indicate separable domains."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polyvore-test", default="data/processed/test.pt")
    parser.add_argument("--catalogue", default="models/catalogue_embeddings.pt")
    parser.add_argument("--output", default="results/domain_shift.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = OutfitDataset(args.polyvore_test)
    positive = dataset.embeddings[dataset.labels == 1]
    positive_masks = dataset.masks[dataset.labels == 1]
    catalogue_payload = torch.load(
        args.catalogue, map_location="cpu", weights_only=False
    )
    if catalogue_payload.get("model_name") != dataset.metadata.get("siglip_model"):
        raise ValueError("Polyvore and catalogue embeddings use different encoders.")
    catalogue_by_slot = {
        slot: torch.stack(
            [item["embedding"].float() for item in catalogue_payload["items"] if item["slot"] == slot]
        ).numpy()
        for slot in SLOTS
    }
    by_slot = {}
    all_polyvore, all_catalogue = [], []
    for slot_index, slot in enumerate(SLOTS):
        polyvore = positive[:, slot_index][positive_masks[:, slot_index]].numpy()
        catalogue = catalogue_by_slot[slot]
        by_slot[slot] = domain_metrics(polyvore, catalogue, seed=args.seed + slot_index)
        all_polyvore.append(polyvore)
        all_catalogue.append(catalogue)
    report = {
        "protocol": (
            "Frozen SigLIP feature-distribution diagnostic using real positive "
            "Polyvore test items and the 300-item product catalogue."
        ),
        "encoder": catalogue_payload.get("model_name"),
        "overall": domain_metrics(
            np.concatenate(all_polyvore), np.concatenate(all_catalogue), args.seed
        ),
        "by_slot": by_slot,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved domain-shift diagnostic to {output_path}")


if __name__ == "__main__":
    main()
