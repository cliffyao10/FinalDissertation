"""Run controlled baselines and architecture ablations on frozen embeddings.

The production checkpoint is never modified. Every neural variant uses the
same split, seed, optimiser settings and validation-AUC selection rule.
"""

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import (
    bootstrap_auc_interval,
    classification_metrics,
    predict,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mean(embeddings, masks):
    """Return one mean frozen-encoder vector per outfit."""
    weights = masks.float().unsqueeze(-1)
    return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class AblationRanker(nn.Module):
    """Compatibility scorer with independently removable structural modules."""

    def __init__(self, config, *, use_slot_embedding=True, use_pairwise=True):
        super().__init__()
        self.config = config
        self.use_slot_embedding = use_slot_embedding
        self.use_pairwise = use_pairwise
        self.image_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
        )
        if use_slot_embedding:
            self.slot_embedding = nn.Embedding(config.number_of_slots, config.slot_dim)
        item_dim = config.hidden_dim + (config.slot_dim if use_slot_embedding else 0)
        scorer_input = item_dim * (2 if use_pairwise else 1)
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, embeddings, mask):
        batch_size, slot_count, _ = embeddings.shape
        if slot_count != self.config.number_of_slots:
            raise ValueError(f"Expected {self.config.number_of_slots} slots.")
        items = self.image_projection(embeddings)
        if self.use_slot_embedding:
            slot_ids = torch.arange(slot_count, device=embeddings.device)
            slot_ids = slot_ids.unsqueeze(0).expand(batch_size, -1)
            items = torch.cat((items, self.slot_embedding(slot_ids)), dim=-1)
        float_mask = mask.float().unsqueeze(-1)
        pooled = (items * float_mask).sum(dim=1)
        pooled = pooled / float_mask.sum(dim=1).clamp_min(1.0)
        if not self.use_pairwise:
            return self.scorer(pooled).squeeze(-1)

        differences, pair_masks = [], []
        for left in range(slot_count):
            for right in range(left + 1, slot_count):
                differences.append(torch.abs(items[:, left] - items[:, right]))
                pair_masks.append(mask[:, left] & mask[:, right])
        pair_values = torch.stack(differences, dim=1)
        pair_mask = torch.stack(pair_masks, dim=1).float().unsqueeze(-1)
        pair_summary = (pair_values * pair_mask).sum(dim=1)
        pair_summary = pair_summary / pair_mask.sum(dim=1).clamp_min(1.0)
        return self.scorer(torch.cat((pooled, pair_summary), dim=-1)).squeeze(-1)


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def make_loader(dataset, batch_size, *, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def validation_auc(model, loader, device):
    labels, probabilities = predict(model, loader, device)
    return classification_metrics(labels, probabilities)["auc"]


def train_variant(name, model_factory, train_data, validation_data, test_data, args, device):
    seed_everything(args.seed)
    model = model_factory().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = make_loader(validation_data, args.batch_size)
    best_auc, best_epoch, best_state = -1.0, 0, None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            loss = loss_function(logits, batch["label"].to(device))
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch["label"])
        auc = validation_auc(model, validation_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(total_loss / len(train_data)),
            "validation_auc": float(auc),
        })
        print(f"{name}: epoch={epoch:02d} validation_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    labels, probabilities = predict(model, make_loader(test_data, args.batch_size), device)
    metrics = classification_metrics(labels, probabilities)
    metrics["auc_95_percent_ci"] = bootstrap_auc_interval(
        labels, probabilities, iterations=args.bootstrap_iterations, seed=args.seed
    )
    checkpoint_path = Path(args.checkpoint_directory) / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "variant": name,
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_auc),
        "seed": args.seed,
    }, checkpoint_path)
    return {
        "kind": "neural_ablation",
        "trainable_parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_auc),
        "test_metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def logistic_baseline(train_data, test_data, args):
    train_features = masked_mean(train_data.embeddings, train_data.masks).numpy()
    test_features = masked_mean(test_data.embeddings, test_data.masks).numpy()
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    classifier.fit(train_features, train_data.labels.numpy().astype(int))
    probabilities = classifier.predict_proba(test_features)[:, 1]
    labels = test_data.labels.numpy().astype(int)
    metrics = classification_metrics(labels, probabilities)
    metrics["auc_95_percent_ci"] = bootstrap_auc_interval(
        labels, probabilities, iterations=args.bootstrap_iterations, seed=args.seed
    )
    return {
        "kind": "mean_pooled_frozen_siglip_logistic_regression",
        "trainable_parameters": int(classifier.coef_.size + classifier.intercept_.size),
        "test_metrics": metrics,
        "configuration": {"C": 1.0, "max_iter": 2000},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--output", default="results/ablation_study.json")
    parser.add_argument("--checkpoint-directory", default="models/ablations")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = OutfitDataset(args.train)
    validation_data = OutfitDataset(args.validation)
    test_data = OutfitDataset(args.test)
    config = ModelConfig(embedding_dim=train_data.embeddings.shape[-1])
    factories = {
        "no_slot": lambda: AblationRanker(config, use_slot_embedding=False, use_pairwise=True),
        "no_pairwise": lambda: AblationRanker(config, use_slot_embedding=True, use_pairwise=False),
        "full_retrained": lambda: CompatibilityRanker(config),
    }
    variants = {"mean_pool_logistic": logistic_baseline(train_data, test_data, args)}
    for name, factory in factories.items():
        variants[name] = train_variant(
            name, factory, train_data, validation_data, test_data, args, device
        )

    report = {
        "purpose": "Controlled comparison on identical frozen SigLIP embeddings; production checkpoint remains unchanged.",
        "datasets": {"train": args.train, "validation": args.validation, "test": args.test},
        "device": device,
        "seed": args.seed,
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
        "variants": variants,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved ablation report to {output_path}")


if __name__ == "__main__":
    main()
