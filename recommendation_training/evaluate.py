"""Evaluate a trained compatibility checkpoint on a held-out outfit split."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset


@torch.no_grad()
def predict(model, loader, device):
    """Return NumPy label and probability arrays for one dataset loader."""

    labels, probabilities = [], []
    model.eval()
    for batch in loader:
        logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def bootstrap_auc_interval(labels, probabilities, iterations=1000, seed=42):
    """Return a deterministic percentile confidence interval for ROC AUC."""

    if iterations <= 0 or len(np.unique(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        scores.append(roc_auc_score(sampled_labels, probabilities[indices]))
    if not scores:
        return None
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return [float(lower), float(upper)]


def classification_metrics(labels, probabilities, threshold=0.5):
    """Calculate discrimination and thresholded classification metrics."""

    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    return {
        "examples": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "threshold": float(threshold),
        "auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) > 1
            else None
        ),
        "average_precision": (
            float(average_precision_score(labels, probabilities))
            if len(np.unique(labels)) > 1
            else None
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": (
            float(true_negative / specificity_denominator)
            if specificity_denominator
            else 0.0
        ),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(labels, predictions)
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--output", default="results/compatibility_test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = OutfitDataset(args.test)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model, checkpoint_metadata = load_checkpoint(args.checkpoint, device)
    labels, probabilities = predict(model, loader, device)
    metrics = classification_metrics(labels, probabilities, args.threshold)
    metrics["auc_95_percent_ci"] = bootstrap_auc_interval(
        labels,
        probabilities,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "test_dataset": str(Path(args.test)),
        "device": device,
        "metrics": metrics,
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_metadata": dataset.metadata,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
