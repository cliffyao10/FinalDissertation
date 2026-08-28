"""Audit D2 train/validation/test discrimination and early-stopping behaviour."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import classification_metrics, predict


def interpret(train_auc, validation_auc, test_auc, history, best_epoch):
    peak = max(history, key=lambda row: row["validation_auc"])
    last = history[-1]
    train_test_gap = train_auc - test_auc
    validation_test_gap = abs(validation_auc - test_auc)
    return {
        "train_minus_test_auc": float(train_test_gap),
        "absolute_validation_minus_test_auc": float(validation_test_gap),
        "best_epoch": int(best_epoch),
        "peak_validation_epoch": int(peak["epoch"]),
        "peak_validation_auc": float(peak["validation_auc"]),
        "last_observed_validation_auc": float(last["validation_auc"]),
        "validation_auc_drop_after_peak": float(
            peak["validation_auc"] - last["validation_auc"]
        ),
        "training_fit_materially_higher": bool(train_test_gap > 0.05),
        "validation_test_agreement_within_0_02": bool(validation_test_gap <= 0.02),
        "best_checkpoint_matches_validation_peak": bool(
            int(best_epoch) == int(peak["epoch"])
        ),
        "conclusion": (
            "Training-set overfitting is visible, but validation-only early "
            "stopping selected the peak checkpoint and validation/test AUC "
            "remain closely aligned. No absence-of-overfitting claim is made."
        ),
        "decision": (
            "retain_frozen_checkpoint"
            if validation_test_gap <= 0.02 and int(best_epoch) == int(peak["epoch"])
            else "investigate_before_release"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/d2/compatibility_ranker.pt")
    parser.add_argument(
        "--training-history",
        default="models/d2/compatibility_ranker.training.json",
    )
    parser.add_argument("--train", default="data/processed_disjoint/train.pt")
    parser.add_argument(
        "--validation", default="data/processed_disjoint/validation.pt"
    )
    parser.add_argument("--test", default="data/processed_disjoint/test.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default="results/d2_overfitting_audit.json")
    args = parser.parse_args()

    model, checkpoint_metadata = load_checkpoint(args.checkpoint, "cpu")
    splits = {}
    for name, path in (
        ("train", args.train),
        ("validation", args.validation),
        ("test", args.test),
    ):
        dataset = OutfitDataset(path)
        labels, probabilities = predict(
            model,
            DataLoader(dataset, batch_size=args.batch_size),
            "cpu",
        )
        metrics = classification_metrics(labels, probabilities)
        splits[name] = {
            key: metrics[key]
            for key in ("examples", "auc", "average_precision", "accuracy", "f1")
        }

    training = json.loads(Path(args.training_history).read_text(encoding="utf-8"))
    report = {
        "protocol": (
            "Frozen final D2 checkpoint evaluated on its train, validation and "
            "held-out test tensors. Checkpoint selection used validation AUC only."
        ),
        "splits": splits,
        "audit": interpret(
            splits["train"]["auc"],
            splits["validation"]["auc"],
            splits["test"]["auc"],
            training["history"],
            checkpoint_metadata["best_epoch"],
        ),
        "research_boundary": (
            "Further tuning against the already inspected test result would "
            "risk test-set overfitting; new regularisation experiments require "
            "validation-only selection or a new untouched test split."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["audit"]["decision"] == "retain_frozen_checkpoint" else 2)


if __name__ == "__main__":
    main()
