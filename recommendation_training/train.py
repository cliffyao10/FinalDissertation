"""Train and evaluate the lightweight compatibility model."""

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    save_checkpoint,
)
from recommendation_training.dataset import OutfitDataset
from recommendation_training.balance import example_weights, group_diagnostics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, dataset=None):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    predictions = [value >= 0.5 for value in probabilities]
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0.0,
    }
    if dataset is not None:
        try:
            metrics["groups"] = group_diagnostics(dataset, labels, probabilities)
        except ValueError as error:
            # Published compatibility files contain balanced labelled rows but
            # do not guarantee the adjacent positive/negative pair structure
            # required by the synthetic-pair diagnostics.
            metrics["groups_unavailable_reason"] = str(error)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="models/compatibility_ranker.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Stop after this many epochs without a validation AUC improvement; 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        choices=("none", "slot", "slot_length"),
        default="none",
        help="Apply smoothed inverse-frequency loss weights to paired groups.",
    )
    parser.add_argument("--balance-exponent", type=float, default=0.5)
    parser.add_argument("--maximum-sample-weight", type=float, default=3.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = OutfitDataset(args.train)
    validation_data = OutfitDataset(args.validation)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    model = CompatibilityRanker(
        ModelConfig(embedding_dim=train_data.embeddings.shape[-1])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    training_weights = example_weights(
        train_data,
        strategy=args.balance,
        exponent=args.balance_exponent,
        maximum=args.maximum_sample_weight,
    )
    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            losses = loss_function(logits, batch["label"].to(device))
            batch_weights = training_weights[batch["index"]].to(device)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        metrics = evaluate(model, validation_loader, device, validation_data)
        mean_loss = total_loss / len(train_data)
        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_auc": float(metrics["auc"]),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d} loss={mean_loss:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_auc={metrics['auc']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_path,
                model,
                extra={
                    "checkpoint_version": 1,
                    "best_epoch": epoch,
                    "best_validation_metrics": {
                        "accuracy": float(metrics["accuracy"]),
                        "auc": float(metrics["auc"]),
                        **(
                            {"groups": metrics["groups"]}
                            if "groups" in metrics
                            else {}
                        ),
                        **(
                            {
                                "groups_unavailable_reason": metrics[
                                    "groups_unavailable_reason"
                                ]
                            }
                            if "groups_unavailable_reason" in metrics
                            else {}
                        ),
                    },
                    "slots": 4,
                    "slot_names": list(train_data.metadata.get("slot_names", [])),
                    "siglip_model": train_data.metadata.get("siglip_model"),
                    "train_examples": len(train_data),
                    "validation_examples": len(validation_data),
                    "train_positive_rate": train_data.positive_rate,
                    "validation_positive_rate": validation_data.positive_rate,
                    "seed": args.seed,
                    "training_config": {
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "patience": args.patience,
                        "balance": args.balance,
                        "balance_exponent": args.balance_exponent,
                        "maximum_sample_weight": args.maximum_sample_weight,
                    },
                },
            )
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    metrics_path = output_path.with_suffix(".training.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_auc": float(best_auc),
                "epochs_completed": len(history),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best model to {output_path} (AUC={best_auc:.4f})")
    print(f"Saved training history to {metrics_path}")


if __name__ == "__main__":
    main()
