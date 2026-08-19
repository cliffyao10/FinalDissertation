"""Test whether an explicit matched-pair ranking loss improves the model."""

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import classification_metrics, predict
from recommendation_training.fashion_model_comparison import ranking_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MatchedPairDataset(Dataset):
    def __init__(self, dataset):
        if len(dataset) % 2:
            raise ValueError("Expected positive/negative pairs.")
        self.dataset = dataset
        for index in range(0, len(dataset), 2):
            if dataset.labels[index].item() != 1 or dataset.labels[index + 1].item() != 0:
                raise ValueError("Pairs must be stored as positive then negative.")

    def __len__(self):
        return len(self.dataset) // 2

    def __getitem__(self, index):
        positive_index = index * 2
        negative_index = positive_index + 1
        return {
            "positive_embeddings": self.dataset.embeddings[positive_index],
            "positive_mask": self.dataset.masks[positive_index],
            "negative_embeddings": self.dataset.embeddings[negative_index],
            "negative_mask": self.dataset.masks[negative_index],
        }


def hybrid_loss(positive_logits, negative_logits, margin=0.2, ranking_weight=0.5):
    logits = torch.cat((positive_logits, negative_logits))
    labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
    classification = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    ranking = torch.relu(margin - positive_logits + negative_logits).mean()
    return classification + ranking_weight * ranking, classification, ranking


def train_run(datasets, seed, args, device):
    seed_everything(seed)
    config = ModelConfig(embedding_dim=datasets["train"].embeddings.shape[-1])
    model = CompatibilityRanker(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MatchedPairDataset(datasets["train"]),
        batch_size=args.pair_batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    best_auc, best_epoch, best_state = -1.0, 0, None
    without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "ranking": 0.0, "pairs": 0}
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            positive_logits = model(
                batch["positive_embeddings"].to(device),
                batch["positive_mask"].to(device),
            )
            negative_logits = model(
                batch["negative_embeddings"].to(device),
                batch["negative_mask"].to(device),
            )
            loss, classification, ranking = hybrid_loss(
                positive_logits,
                negative_logits,
                margin=args.margin,
                ranking_weight=args.ranking_weight,
            )
            loss.backward()
            optimiser.step()
            pairs = len(positive_logits)
            totals["loss"] += loss.item() * pairs
            totals["classification"] += classification.item() * pairs
            totals["ranking"] += ranking.item() * pairs
            totals["pairs"] += pairs
        labels, probabilities = predict(model, validation_loader, device)
        validation_auc = classification_metrics(labels, probabilities)["auc"]
        history.append({
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["pairs"],
            "classification_loss": totals["classification"] / totals["pairs"],
            "ranking_loss": totals["ranking"] / totals["pairs"],
            "validation_auc": validation_auc,
        })
        print(f"seed={seed} epoch={epoch:02d} validation_auc={validation_auc:.4f}")
        if validation_auc > best_auc:
            best_auc, best_epoch = validation_auc, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
            if args.patience and without_improvement >= args.patience:
                break
    model.load_state_dict(best_state)
    model.to(device).eval()
    labels, probabilities = predict(
        model, DataLoader(datasets["test"], batch_size=args.batch_size), device
    )
    metrics = classification_metrics(labels, probabilities)
    metrics.update(
        ranking_metrics(
            model,
            datasets["test"],
            device,
            seed=args.fitb_seed,
            batch_size=args.batch_size,
        )
    )
    checkpoint_path = Path(args.checkpoint_directory) / f"hybrid_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": vars(config),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "objective": {
            "classification": "binary_cross_entropy_with_logits",
            "ranking": "matched_pair_margin_ranking",
            "margin": args.margin,
            "ranking_weight": args.ranking_weight,
        },
    }, checkpoint_path)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "test_metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def metric_summary(runs):
    names = (
        "auc",
        "average_precision",
        "balanced_accuracy",
        "pair_ranking_accuracy",
        "fitb4_accuracy",
        "fitb4_mean_reciprocal_rank",
        "fitb4_ndcg",
    )
    return {
        name: {
            "values": [run["test_metrics"][name] for run in runs],
            "mean": float(np.mean([run["test_metrics"][name] for run in runs])),
            "sample_standard_deviation": float(
                np.std([run["test_metrics"][name] for run in runs], ddof=1)
            ),
        }
        for name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--fitb-seed", type=int, default=2026)
    parser.add_argument("--checkpoint-directory", default="models/ranking_objective")
    parser.add_argument("--output", default="results/ranking_objective_study.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        "train": OutfitDataset(args.train),
        "validation": OutfitDataset(args.validation),
        "test": OutfitDataset(args.test),
    }
    runs = [train_run(datasets, seed, args, device) for seed in args.seeds]
    report = {
        "protocol": (
            "The proposed architecture is unchanged. Training adds a matched-pair "
            "margin loss to BCE and retains validation-AUC model selection."
        ),
        "configuration": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "ranking_weight": args.ranking_weight,
        },
        "runs": runs,
        "summary": metric_summary(runs),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved ranking-objective study to {output_path}")


if __name__ == "__main__":
    main()
