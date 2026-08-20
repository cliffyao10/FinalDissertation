"""Group-aware weighting and diagnostics for paired outfit examples."""

from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from recommendation_training.dataset import SLOTS


def infer_pair_groups(dataset):
    """Assign both examples in each pair to its changed slot and outfit length."""

    if len(dataset) % 2:
        raise ValueError("Expected an even number of paired examples.")
    groups = []
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
        groups.append(
            {
                "positive_index": positive_index,
                "negative_index": negative_index,
                "slot": changed_slots[0],
                "present_slots": int(positive_mask.sum().item()),
            }
        )
    return groups


def example_weights(dataset, strategy="none", exponent=0.5, maximum=3.0):
    """Return mean-one smoothed inverse-frequency weights for paired examples.

    ``slot`` balances the changed clothing position. ``slot_length`` balances
    the joint changed-position/outfit-length groups. The exponent and cap keep
    very small groups from dominating optimisation.
    """

    if strategy not in {"none", "slot", "slot_length"}:
        raise ValueError(f"Unknown balance strategy: {strategy}")
    weights = torch.ones(len(dataset), dtype=torch.float32)
    if strategy == "none":
        return weights
    groups = infer_pair_groups(dataset)
    keys = [
        (group["slot"],)
        if strategy == "slot"
        else (group["slot"], group["present_slots"])
        for group in groups
    ]
    counts = Counter(keys)
    target = len(groups) / len(counts)
    for group, key in zip(groups, keys):
        value = (target / counts[key]) ** exponent
        value = min(float(maximum), value) if maximum else value
        weights[group["positive_index"]] = value
        weights[group["negative_index"]] = value
    return weights / weights.mean()


def _safe_auc(labels, probabilities):
    return (
        float(roc_auc_score(labels, probabilities))
        if len(labels) and len(np.unique(labels)) > 1
        else None
    )


def group_diagnostics(dataset, labels, probabilities):
    """Report disparity across changed slots and three/four-piece outfits."""

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    groups = infer_pair_groups(dataset)
    slot_metrics = {}
    for slot_index, slot_name in enumerate(SLOTS):
        selected = [group for group in groups if group["slot"] == slot_index]
        indices = np.asarray(
            [index for group in selected for index in (group["positive_index"], group["negative_index"])],
            dtype=np.int64,
        )
        pair_accuracy = (
            float(np.mean([
                probabilities[group["positive_index"]]
                > probabilities[group["negative_index"]]
                for group in selected
            ]))
            if selected
            else None
        )
        slot_metrics[slot_name] = {
            "pairs": len(selected),
            "auc": _safe_auc(labels[indices], probabilities[indices]) if len(indices) else None,
            "pair_ranking_accuracy": pair_accuracy,
        }
    length_metrics = {}
    for length in sorted({group["present_slots"] for group in groups}):
        selected = [group for group in groups if group["present_slots"] == length]
        indices = np.asarray(
            [index for group in selected for index in (group["positive_index"], group["negative_index"])],
            dtype=np.int64,
        )
        length_metrics[str(length)] = {
            "pairs": len(selected),
            "auc": _safe_auc(labels[indices], probabilities[indices]),
        }
    slot_aucs = [value["auc"] for value in slot_metrics.values() if value["auc"] is not None]
    length_aucs = [value["auc"] for value in length_metrics.values() if value["auc"] is not None]
    return {
        "by_replaced_slot": slot_metrics,
        "by_present_slot_count": length_metrics,
        "macro_slot_auc": float(np.mean(slot_aucs)),
        "worst_slot_auc": float(min(slot_aucs)),
        "slot_auc_gap": float(max(slot_aucs) - min(slot_aucs)),
        "macro_length_auc": float(np.mean(length_aucs)),
        "length_auc_gap": float(max(length_aucs) - min(length_aucs)),
    }
