"""Benchmark exact abstract-outfit search on the local runtime."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from recommendation_training.dataset import SLOTS
from recommendation_training.ranker import OutfitCandidateRanker, choose_diverse_pair
from src.trained_recommender import _audience_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--prototypes", default="models/polyvore_abstract_prototypes.pt")
    parser.add_argument("--output", default="results/abstract_search_benchmark.json")
    parser.add_argument("--style", default="Casual")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--audience", default="womenswear")
    args = parser.parse_args()

    payload = torch.load(args.prototypes, map_location="cpu", weights_only=False)
    all_items = payload["items"]
    audience_items = _audience_filter(all_items, args.audience)
    styled = [item for item in audience_items if item.get("style") == args.style]
    candidates = styled if {item["slot"] for item in styled} == set(SLOTS) else audience_items
    by_slot = {slot: [item for item in candidates if item["slot"] == slot] for slot in SLOTS}
    ranker = OutfitCandidateRanker(args.checkpoint, device=args.device)

    runs = []
    for input_slot in SLOTS:
        fixed = by_slot[input_slot][0]
        candidate_slots = {
            slot: slot_items
            for slot, slot_items in by_slot.items()
            if slot != input_slot
        }
        for repeat in range(args.repeats):
            ranked = ranker.rank_exact(
                {input_slot: fixed},
                candidate_slots,
                top_k=args.top_k,
                batch_size=args.batch_size,
                maximum_combinations=5_000_000,
            )
            primary, alternative = choose_diverse_pair(ranked)
            primary_attributes = {
                item["slot"]: (item.get("type"), item.get("colour"))
                for item in primary["items"]
            }
            alternative_changes = sum(
                primary_attributes.get(item["slot"])
                != (item.get("type"), item.get("colour"))
                for item in alternative["items"]
                if item["slot"] in primary_attributes
            )
            visible_signatures = {
                tuple(
                    sorted(
                        (
                            item["slot"],
                            item.get("type"),
                            item.get("colour"),
                        )
                        for item in candidate["items"]
                    )
                )
                for candidate in ranked
            }
            runs.append(
                {
                    "input_slot": input_slot,
                    "repeat": repeat + 1,
                    **ranker.last_search_stats,
                    "unique_visible_outfits_in_top_k": len(visible_signatures),
                    "primary_alternative_attribute_changes": alternative_changes,
                    "primary_score": primary["compatibility_score"],
                    "alternative_score": alternative["compatibility_score"],
                }
            )

    elapsed = [run["elapsed_seconds"] for run in runs]
    report = {
        "method": "batched_exact_global_top_k_over_abstract_states",
        "checkpoint": str(args.checkpoint),
        "prototype_artifact": str(args.prototypes),
        "prototype_artifact_bytes": Path(args.prototypes).stat().st_size,
        "prototype_count": len(all_items),
        "audience": args.audience,
        "audience_prototype_count": len(audience_items),
        "style": args.style,
        "candidate_counts_by_slot": Counter(item["slot"] for item in candidates),
        "device": str(ranker.device),
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "elapsed_seconds": {
            "median": statistics.median(elapsed),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "top_k_diversity": {
            "minimum_unique_visible_outfits": min(
                run["unique_visible_outfits_in_top_k"] for run in runs
            ),
            "minimum_primary_alternative_attribute_changes": min(
                run["primary_alternative_attribute_changes"] for run in runs
            ),
        },
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
