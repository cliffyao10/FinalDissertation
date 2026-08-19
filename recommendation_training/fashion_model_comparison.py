"""Controlled comparison with representative fashion compatibility models.

This experiment deliberately keeps the frozen SigLIP item embeddings, data
splits, optimiser, early-stopping rule and evaluation examples fixed.  The
architectures represent three established modelling families rather than
claiming byte-for-byte reproductions of the original papers:

* a slot-ordered bidirectional LSTM;
* type-aware, slot-pair-specific compatibility projections; and
* a permutation-aware Transformer with a learned outfit token.

The report separates results on this project's processed split from numbers
reported by prior work.  They must not be presented as directly comparable.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import OutfitDataset, SLOTS
from recommendation_training.evaluate import (
    bootstrap_auc_interval,
    classification_metrics,
    predict,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BiLSTMCompatibilityBaseline(nn.Module):
    """Sequence baseline inspired by Han et al.'s Bi-LSTM outfit model."""

    def __init__(self, embedding_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, mask):
        projected = self.projection(embeddings)
        sequences = [projected[index][mask[index]] for index in range(len(projected))]
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences], device="cpu", dtype=torch.long
        )
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.scorer(representation).squeeze(-1)


class TypeAwarePairwiseBaseline(nn.Module):
    """Adapt type-specific embeddings to the project's six fixed slot pairs."""

    def __init__(self, embedding_dim, pair_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.pairs = tuple(
            (left, right)
            for left in range(len(SLOTS))
            for right in range(left + 1, len(SLOTS))
        )
        self.left_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, pair_dim) for _ in self.pairs]
        )
        self.right_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, pair_dim) for _ in self.pairs]
        )
        self.scorer = nn.Sequential(
            nn.Linear(len(self.pairs) * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, mask):
        similarities, present = [], []
        for index, (left, right) in enumerate(self.pairs):
            left_value = nn.functional.normalize(
                self.left_projections[index](embeddings[:, left]), dim=-1
            )
            right_value = nn.functional.normalize(
                self.right_projections[index](embeddings[:, right]), dim=-1
            )
            pair_present = mask[:, left] & mask[:, right]
            similarity = (left_value * right_value).sum(dim=-1)
            similarities.append(similarity * pair_present.float())
            present.append(pair_present.float())
        features = torch.cat(
            (torch.stack(similarities, dim=-1), torch.stack(present, dim=-1)), dim=-1
        )
        return self.scorer(features).squeeze(-1)


class SetTransformerCompatibilityBaseline(nn.Module):
    """Small outfit-token Transformer representing higher-order set modelling."""

    def __init__(
        self,
        embedding_dim,
        model_dim=128,
        heads=4,
        layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, model_dim),
            nn.GELU(),
        )
        self.slot_embedding = nn.Embedding(len(SLOTS), model_dim)
        self.outfit_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.outfit_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, embeddings, mask):
        batch_size, slot_count, _ = embeddings.shape
        slot_ids = torch.arange(slot_count, device=embeddings.device)
        items = self.projection(embeddings) + self.slot_embedding(slot_ids).unsqueeze(0)
        token = self.outfit_token.expand(batch_size, -1, -1)
        sequence = torch.cat((token, items), dim=1)
        token_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        present = torch.cat((token_mask, mask), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~present)
        return self.scorer(encoded[:, 0]).squeeze(-1)


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def make_loader(dataset, batch_size, *, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def infer_matched_replacement_groups(dataset):
    """Recover positive/negative pairs emitted consecutively by build_examples."""

    groups = []
    if len(dataset) % 2:
        raise ValueError("Expected an even number of paired examples.")
    for positive_index in range(0, len(dataset), 2):
        negative_index = positive_index + 1
        if dataset.labels[positive_index].item() != 1.0:
            raise ValueError("Expected each pair to start with a positive example.")
        if dataset.labels[negative_index].item() != 0.0:
            raise ValueError("Expected each pair to end with a negative example.")
        positive_mask = dataset.masks[positive_index]
        negative_mask = dataset.masks[negative_index]
        if not torch.equal(positive_mask, negative_mask):
            raise ValueError("Paired examples must have identical slot masks.")
        changed = (
            (dataset.embeddings[positive_index] - dataset.embeddings[negative_index])
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
            & positive_mask
        )
        changed_slots = torch.where(changed)[0].tolist()
        if len(changed_slots) != 1:
            raise ValueError("Each negative must replace exactly one present slot.")
        slot = changed_slots[0]
        groups.append(
            {
                "positive_index": positive_index,
                "negative_index": negative_index,
                "slot": slot,
                "replacement": dataset.embeddings[negative_index, slot].clone(),
            }
        )
    return groups


@torch.no_grad()
def score_tensors(model, embeddings, masks, device, batch_size=256):
    model.eval()
    scores = []
    for start in range(0, len(embeddings), batch_size):
        stop = start + batch_size
        logits = model(embeddings[start:stop].to(device), masks[start:stop].to(device))
        scores.extend(logits.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def ranking_metrics(model, dataset, device, seed=42, batch_size=256):
    """Evaluate matched ranking and a deterministic four-choice completion task."""

    groups = infer_matched_replacement_groups(dataset)
    all_scores = score_tensors(
        model, dataset.embeddings, dataset.masks, device, batch_size=batch_size
    )
    pair_wins = [
        all_scores[group["positive_index"]] > all_scores[group["negative_index"]]
        for group in groups
    ]

    pools = defaultdict(list)
    for group_index, group in enumerate(groups):
        pools[group["slot"]].append((group_index, group["replacement"]))
    rng = random.Random(seed)
    completion_embeddings, completion_masks, answer_offsets = [], [], []
    eligible_groups = 0
    for group_index, group in enumerate(groups):
        alternatives = [
            value for index, value in pools[group["slot"]] if index != group_index
        ]
        if len(alternatives) < 2:
            continue
        positive = dataset.embeddings[group["positive_index"]]
        candidates = [positive]
        for replacement in [group["replacement"], *rng.sample(alternatives, 2)]:
            corrupted = positive.clone()
            corrupted[group["slot"]] = replacement
            candidates.append(corrupted)
        order = list(range(4))
        rng.shuffle(order)
        answer_offsets.append(order.index(0))
        completion_embeddings.extend(candidates[index] for index in order)
        completion_masks.extend(dataset.masks[group["positive_index"]] for _ in order)
        eligible_groups += 1

    ranks = []
    if completion_embeddings:
        fitb_scores = score_tensors(
            model,
            torch.stack(completion_embeddings),
            torch.stack(completion_masks),
            device,
            batch_size=batch_size,
        ).reshape(-1, 4)
        for row, answer in zip(fitb_scores, answer_offsets):
            ranks.append(1 + int(np.sum(row[np.arange(4) != answer] >= row[answer])))
    return {
        "matched_pairs": len(groups),
        "pair_ranking_accuracy": float(np.mean(pair_wins)) if pair_wins else None,
        "fitb4_examples": eligible_groups,
        "fitb4_accuracy": float(np.mean(np.asarray(ranks) == 1)) if ranks else None,
        "fitb4_mean_reciprocal_rank": (
            float(np.mean(1.0 / np.asarray(ranks))) if ranks else None
        ),
        "fitb4_ndcg": (
            float(np.mean(1.0 / np.log2(np.asarray(ranks) + 1))) if ranks else None
        ),
        "fitb4_protocol": (
            "One true completion and three same-slot replacements. One replacement is "
            "the matched negative; two are sampled deterministically from other test "
            "pairs. This is a project-specific diagnostic, not the official Polyvore FITB."
        ),
    }


def train_and_evaluate(name, factory, datasets, args, seed, device):
    seed_everything(seed)
    model = factory().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_loader = make_loader(
        datasets["train"], args.batch_size, shuffle=True, seed=seed
    )
    validation_loader = make_loader(datasets["validation"], args.batch_size)
    best_auc, best_epoch, best_state = -1.0, 0, None
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
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
        labels, probabilities = predict(model, validation_loader, device)
        validation_auc = classification_metrics(labels, probabilities)["auc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(total_loss / len(datasets["train"])),
                "validation_auc": float(validation_auc),
            }
        )
        print(
            f"seed={seed} model={name} epoch={epoch:02d} "
            f"validation_auc={validation_auc:.4f}"
        )
        if validation_auc > best_auc:
            best_auc, best_epoch = validation_auc, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device).eval()
    test_loader = make_loader(datasets["test"], args.batch_size)
    labels, probabilities = predict(model, test_loader, device)
    metrics = classification_metrics(labels, probabilities)
    metrics["auc_95_percent_ci"] = bootstrap_auc_interval(
        labels,
        probabilities,
        iterations=args.bootstrap_iterations,
        seed=seed,
    )
    metrics.update(
        ranking_metrics(
            model,
            datasets["test"],
            device,
            seed=args.fitb_seed,
            batch_size=args.batch_size,
        )
    )
    checkpoint_path = Path(args.checkpoint_directory) / f"{name}_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_auc": float(best_auc),
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "trainable_parameters": parameter_count(model),
        "training_seconds": float(training_seconds),
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_auc),
        "test_metrics": metrics,
        "test_probabilities": probabilities.tolist(),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def summarise(runs):
    metric_names = (
        "auc",
        "average_precision",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
        "pair_ranking_accuracy",
        "fitb4_accuracy",
        "fitb4_mean_reciprocal_rank",
        "fitb4_ndcg",
    )
    summary = {}
    for model_name, model_runs in runs.items():
        summary[model_name] = {
            "trainable_parameters": model_runs[0]["trainable_parameters"],
            "training_seconds_mean": float(
                np.mean([run["training_seconds"] for run in model_runs])
            ),
            "metrics": {},
        }
        for metric_name in metric_names:
            values = [run["test_metrics"][metric_name] for run in model_runs]
            summary[model_name]["metrics"][metric_name] = {
                "values": values,
                "mean": float(np.mean(values)),
                "sample_standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }
    return summary


def paired_bootstrap_auc_comparisons(
    runs, labels, proposed_name, iterations=2000, seed=2026
):
    """Compare seed-ensembled models on the same sampled test examples."""

    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    mean_scores = {
        name: np.mean(
            np.asarray([run["test_probabilities"] for run in model_runs]), axis=0
        )
        for name, model_runs in runs.items()
    }
    proposed_scores = mean_scores[proposed_name]
    rng = np.random.default_rng(seed)
    comparisons = {}
    for baseline_name, baseline_scores in mean_scores.items():
        if baseline_name == proposed_name:
            continue
        observed = roc_auc_score(labels, proposed_scores) - roc_auc_score(
            labels, baseline_scores
        )
        differences = []
        for _ in range(iterations):
            indices = rng.integers(0, len(labels), size=len(labels))
            sampled_labels = labels[indices]
            if len(np.unique(sampled_labels)) < 2:
                continue
            differences.append(
                roc_auc_score(sampled_labels, proposed_scores[indices])
                - roc_auc_score(sampled_labels, baseline_scores[indices])
            )
        lower, upper = np.percentile(differences, [2.5, 97.5])
        comparisons[baseline_name] = {
            "proposed_minus_baseline_auc": float(observed),
            "paired_bootstrap_95_percent_ci": [float(lower), float(upper)],
            "bootstrap_iterations": len(differences),
            "interpretation": (
                "The interval estimates the AUC difference on this processed test "
                "set using mean predictions across training seeds. It is not a "
                "comparison against the official published Polyvore benchmark."
            ),
        }
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--output", default="results/fashion_model_comparison.json")
    parser.add_argument(
        "--checkpoint-directory", default="models/fashion_model_comparison"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--fitb-seed", type=int, default=2026)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        "train": OutfitDataset(args.train),
        "validation": OutfitDataset(args.validation),
        "test": OutfitDataset(args.test),
    }
    embedding_dim = datasets["train"].embeddings.shape[-1]
    config = ModelConfig(embedding_dim=embedding_dim)
    factories = {
        "bilstm": lambda: BiLSTMCompatibilityBaseline(embedding_dim),
        "type_aware_pairwise": lambda: TypeAwarePairwiseBaseline(embedding_dim),
        "set_transformer": lambda: SetTransformerCompatibilityBaseline(embedding_dim),
        "lightweight_pairwise_proposed": lambda: CompatibilityRanker(config),
    }
    runs = {name: [] for name in factories}
    for seed in args.seeds:
        for name, factory in factories.items():
            runs[name].append(
                train_and_evaluate(name, factory, datasets, args, seed, device)
            )

    report = {
        "study_design": {
            "purpose": (
                "Controlled architecture comparison on identical frozen SigLIP "
                "embeddings and identical processed Polyvore examples."
            ),
            "scope_warning": (
                "These are architecture-family baselines adapted to this project's "
                "four-slot tensors, not exact reproductions of published systems. "
                "Published benchmark numbers are not directly comparable."
            ),
            "controlled_variables": [
                "frozen item embeddings",
                "train/validation/test split",
                "positive and negative examples",
                "optimizer and learning rate",
                "batch size and maximum epochs",
                "validation-AUC checkpoint selection",
                "FITB-4 candidate construction",
            ],
            "datasets": {
                name: {
                    "path": getattr(args, name),
                    "examples": len(dataset),
                    "positive_rate": dataset.positive_rate,
                    "metadata": dataset.metadata,
                }
                for name, dataset in datasets.items()
            },
            "device": device,
            "seeds": args.seeds,
            "training_configuration": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "fitb_seed": args.fitb_seed,
            },
        },
        "model_definitions": {
            "bilstm": "Slot-ordered bidirectional LSTM sequence baseline.",
            "type_aware_pairwise": (
                "Six slot-pair-specific projection pairs followed by cosine "
                "compatibility aggregation."
            ),
            "set_transformer": (
                "Two-layer, four-head Transformer with slot embeddings and a "
                "learned outfit token."
            ),
            "lightweight_pairwise_proposed": (
                "Proposed pooled and absolute-pair-difference compatibility head."
            ),
        },
        "runs": runs,
        "summary": summarise(runs),
        "paired_auc_comparisons": paired_bootstrap_auc_comparisons(
            runs,
            datasets["test"].labels.numpy().astype(int),
            "lightweight_pairwise_proposed",
            iterations=args.bootstrap_iterations,
            seed=args.fitb_seed,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved controlled comparison to {output_path}")


if __name__ == "__main__":
    main()
