"""Train and evaluate the lightweight compatibility model."""

import argparse
import random
from pathlib import Path

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    predictions = [value >= 0.5 for value in probabilities]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="models/compatibility_ranker.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            loss = loss_function(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        metrics = evaluate(model, validation_loader, device)
        mean_loss = total_loss / len(train_data)
        print(
            f"epoch={epoch:02d} loss={mean_loss:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_auc={metrics['auc']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            save_checkpoint(
                output_path,
                model,
                extra={"best_validation_metrics": metrics, "slots": 4},
            )
    print(f"Saved best model to {output_path} (AUC={best_auc:.4f})")


if __name__ == "__main__":
    main()
