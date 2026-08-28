"""Evaluate all controlled models on the official Polyvore FITB questions.

Only questions representable by the product's four-slot taxonomy and available
in the frozen embedding cache are retained. Coverage and every exclusion reason
are reported so that the subset cannot be mistaken for the full benchmark.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import SLOTS
from recommendation_training.fashion_model_comparison import (
    BiLSTMCompatibilityBaseline,
    SetTransformerCompatibilityBaseline,
    TypeAwarePairwiseBaseline,
    score_tensors,
)
from recommendation_training.prepare_polyvore import CATEGORY_TO_SLOT
from src.audience import audience_for_category_id


def stable_item_id(item):
    match = re.search(r"[?&]tid=(\d+)", str(item.get("image", "")))
    return match.group(1) if match else None


def reference_set_id(reference):
    """Return the outfit id from an official ``set_id_item_index`` key."""

    value = str(reference)
    return value.rsplit("_", 1)[0] if "_" in value else None


def load_reference_lookup(metadata_path, item_metadata_path=None):
    """Map official ``set_id_item_index`` references to cached item metadata."""

    item_metadata = (
        json.loads(Path(item_metadata_path).read_text(encoding="utf-8"))
        if item_metadata_path
        else {}
    )
    lookup = {}
    for outfit in json.loads(Path(metadata_path).read_text(encoding="utf-8")):
        set_id = str(outfit["set_id"])
        for item in outfit.get("items", []):
            item_id = str(item.get("item_id") or stable_item_id(item) or "")
            category_id = item.get("categoryid")
            if category_id is None:
                category_id = item_metadata.get(item_id, {}).get("category_id")
            lookup[f"{set_id}_{item['index']}"] = {
                "item_id": item_id or None,
                "slot": (
                    CATEGORY_TO_SLOT.get(int(category_id))
                    if category_id not in (None, "")
                    else None
                ),
                "audience": (
                    audience_for_category_id(category_id)
                    if category_id not in (None, "")
                    and CATEGORY_TO_SLOT.get(int(category_id)) is not None
                    else None
                ),
            }
    return lookup


def prepare_official_fitb(
    metadata_path,
    questions_path,
    cache_path,
    minimum_slots=3,
    item_metadata_path=None,
):
    """Build fixed four-choice tensors and retain transparent coverage counts."""

    references = load_reference_lookup(metadata_path, item_metadata_path)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    embedding_by_id = dict(zip(cache["item_ids"], cache["embeddings"].float()))
    dimension = cache["embeddings"].shape[-1]
    slot_to_index = {slot: index for index, slot in enumerate(SLOTS)}
    examples, masks, retained_question_indices, correct_answer_indices = [], [], [], []
    question_audiences = []
    skipped = Counter()

    for question_index, question in enumerate(questions):
        question_set_ids = {
            reference_set_id(reference) for reference in question["question"]
        }
        question_set_ids.discard(None)
        if len(question_set_ids) != 1:
            skipped["ambiguous_question_set"] += 1
            continue
        question_set_id = next(iter(question_set_ids))
        matching_answers = [
            index
            for index, reference in enumerate(question["answers"])
            if reference_set_id(reference) == question_set_id
        ]
        # Some small third-party/synthetic fixtures preserve the historical
        # nondisjoint convention where every answer key shares the same set
        # prefix and the correct answer is first.  Official disjoint questions
        # have exactly one set-id match and take the branch above.
        if len(matching_answers) == len(question["answers"]):
            matching_answers = [0]
        if len(matching_answers) != 1:
            skipped["ambiguous_correct_answer"] += 1
            continue
        answer_records = [references.get(reference) for reference in question["answers"]]
        if any(record is None for record in answer_records):
            skipped["unknown_answer_reference"] += 1
            continue
        answer_slots = [record["slot"] for record in answer_records]
        if any(slot is None for slot in answer_slots):
            skipped["unsupported_answer_slot"] += 1
            continue
        if len(set(answer_slots)) != 1:
            skipped["answer_slot_mismatch"] += 1
            continue
        if any(
            record["item_id"] not in embedding_by_id
            for record in answer_records
        ):
            skipped["missing_answer_embedding"] += 1
            continue

        base = torch.zeros(len(SLOTS), dimension)
        mask = torch.zeros(len(SLOTS), dtype=torch.bool)
        invalid_reason = None
        for reference in question["question"]:
            record = references.get(reference)
            if record is None:
                invalid_reason = "unknown_question_reference"
                break
            if record["slot"] is None:
                continue
            if record["item_id"] not in embedding_by_id:
                invalid_reason = "missing_question_embedding"
                break
            slot_index = slot_to_index[record["slot"]]
            if mask[slot_index]:
                invalid_reason = "duplicate_question_slot"
                break
            base[slot_index] = embedding_by_id[record["item_id"]]
            mask[slot_index] = True
        if invalid_reason:
            skipped[invalid_reason] += 1
            continue
        answer_slot_indices = [slot_to_index[slot] for slot in answer_slots]
        if any(mask[index] for index in answer_slot_indices):
            skipped["answer_slot_already_present"] += 1
            continue
        if int(mask.sum()) + 1 < minimum_slots:
            skipped["fewer_than_minimum_supported_slots"] += 1
            continue

        for answer, target_index in zip(answer_records, answer_slot_indices):
            candidate = base.clone()
            candidate[target_index] = embedding_by_id[answer["item_id"]]
            candidate_mask = mask.clone()
            candidate_mask[target_index] = True
            examples.append(candidate)
            masks.append(candidate_mask)
        retained_question_indices.append(question_index)
        correct_answer_indices.append(matching_answers[0])
        question_audiences.append(
            answer_records[matching_answers[0]].get("audience")
        )

    return {
        "embeddings": (
            torch.stack(examples)
            if examples
            else torch.empty(0, len(SLOTS), dimension)
        ),
        "masks": (
            torch.stack(masks)
            if masks
            else torch.empty(0, len(SLOTS), dtype=torch.bool)
        ),
        "retained_question_indices": retained_question_indices,
        "correct_answer_indices": correct_answer_indices,
        "question_audiences": question_audiences,
        "total_questions": len(questions),
        "retained_questions": len(retained_question_indices),
        "skipped": dict(sorted(skipped.items())),
        "siglip_model": cache.get("model_name"),
    }


def fitb_metrics(scores, correct_answer_indices=None):
    """Rank the answer belonging to the question outfit.

    The nondisjoint file commonly puts the correct answer first, whereas the
    official disjoint file shuffles answer order.  Callers should therefore
    pass the answer indices derived from the shared outfit/set identifier.
    """

    rows = np.asarray(scores, dtype=np.float64).reshape(-1, 4)
    if correct_answer_indices is None:
        correct_answer_indices = np.zeros(len(rows), dtype=np.int64)
    correct_answer_indices = np.asarray(correct_answer_indices, dtype=np.int64)
    if correct_answer_indices.shape != (len(rows),):
        raise ValueError("FITB needs one correct-answer index per question.")
    if np.any((correct_answer_indices < 0) | (correct_answer_indices >= 4)):
        raise ValueError("FITB correct-answer indices must be between 0 and 3.")
    correct_scores = rows[np.arange(len(rows)), correct_answer_indices]
    competitors = np.arange(4)[None, :] != correct_answer_indices[:, None]
    ranks = 1 + np.sum((rows >= correct_scores[:, None]) & competitors, axis=1)
    return {
        "questions": int(len(rows)),
        "accuracy": float(np.mean(ranks == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "ndcg": float(np.mean(1.0 / np.log2(ranks + 1))),
        "chance_accuracy": 0.25,
    }


def model_factories(embedding_dim):
    config = ModelConfig(embedding_dim=embedding_dim)
    return {
        "bilstm": lambda: BiLSTMCompatibilityBaseline(embedding_dim),
        "type_aware_pairwise": lambda: TypeAwarePairwiseBaseline(embedding_dim),
        "set_transformer": lambda: SetTransformerCompatibilityBaseline(embedding_dim),
        "lightweight_pairwise_proposed": lambda: CompatibilityRanker(config),
    }


def summarise(runs):
    report = {}
    for model_name, model_runs in runs.items():
        report[model_name] = {}
        for metric in ("accuracy", "mean_reciprocal_rank", "ndcg"):
            values = [run[metric] for run in model_runs]
            report[model_name][metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
            }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/polyvore/test_no_dup.json")
    parser.add_argument("--item-metadata")
    parser.add_argument("--questions", default="data/polyvore/fill_in_blank_test.json")
    parser.add_argument("--cache", default="data/processed/fitb_item_embeddings.pt")
    parser.add_argument(
        "--checkpoint-directory", default="models/fashion_model_comparison"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-slots", type=int, default=3)
    parser.add_argument("--output", default="results/official_fitb_comparison.json")
    args = parser.parse_args()

    payload = prepare_official_fitb(
        args.metadata,
        args.questions,
        args.cache,
        minimum_slots=args.minimum_slots,
        item_metadata_path=args.item_metadata,
    )
    coverage = {
        "total_questions": payload["total_questions"],
        "retained_questions": payload["retained_questions"],
        "retained_fraction": payload["retained_questions"]
        / payload["total_questions"],
        "skipped": payload["skipped"],
        "selection_bias_warning": (
            "This filtered result is not numerically interchangeable with a "
            "full-dataset published FITB score."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not payload["retained_questions"]:
        report = {
            "status": "unavailable_with_current_image_archive",
            "protocol": (
                "Official questions were parsed, but no question had four "
                "supported answer images available in the local secondary-data "
                "archive. No model score is reported."
            ),
            "coverage": coverage,
            "runs": {},
            "summary": {},
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved official FITB availability audit to {output_path}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    factories = model_factories(payload["embeddings"].shape[-1])
    runs = {name: [] for name in factories}
    for model_name, factory in factories.items():
        for seed in args.seeds:
            checkpoint_path = (
                Path(args.checkpoint_directory) / f"{model_name}_seed{seed}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = factory().to(device)
            model.load_state_dict(checkpoint["model_state"])
            scores = score_tensors(
                model,
                payload["embeddings"],
                payload["masks"],
                device,
                batch_size=args.batch_size,
            )
            metrics = fitb_metrics(scores, payload["correct_answer_indices"])
            metrics.update({"seed": seed, "checkpoint": str(checkpoint_path)})
            runs[model_name].append(metrics)

    report = {
        "protocol": (
            "Official Polyvore fill_in_blank_test.json questions, restricted to "
            "questions representable by the project's four-slot taxonomy with "
            "available frozen embeddings. The correct answer is identified by "
            "matching the answer set_id to the question outfit; disjoint answer "
            "order is shuffled."
        ),
        "coverage": coverage,
        "device": device,
        "siglip_model": payload["siglip_model"],
        "minimum_slots": args.minimum_slots,
        "seeds": args.seeds,
        "runs": runs,
        "summary": summarise(runs),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "summary": report["summary"]}, indent=2))
    print(f"Saved official FITB subset evaluation to {output_path}")


if __name__ == "__main__":
    main()
