"""Compare Cove v3 search with the prior rule and catalogue pipelines.

The compatibility checkpoint is intentionally held constant between v2 and
v3.  This experiment therefore separates model discrimination from candidate
source and search quality; a higher model score is not labelled as human
preference accuracy.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from recommendation_training.dataset import SLOTS
from recommendation_training.ranker import OutfitCandidateRanker, choose_diverse_pair
from src.garment_taxonomy import broad_garment_category
from src.recommendation import RANKER as RULE_RANKER
from src.trained_recommender import (
    _audience_filter,
    _shortlist_candidates,
    _style_filter,
    _weather_filter,
)


STYLES = (
    "Casual",
    "Outdoor",
    "Sporty",
    "Formal",
    "Business",
    "Streetwear",
    "Minimalist",
    "Party",
    "Beachwear",
)

INPUT_COLOURS = (
    "Black", "White", "Grey", "Red", "Orange", "Yellow",
    "Green", "Blue", "Purple", "Pink", "Brown", "Beige",
)

WEATHER_SCENARIOS = {
    "mild": {"feels_like": 20, "rain_probability": 0, "condition": "Clear"},
    "hot": {"feels_like": 32, "rain_probability": 0, "condition": "Clear"},
    "cold": {"feels_like": 2, "rain_probability": 0, "condition": "Clear"},
    "rain": {"feels_like": 12, "rain_probability": 80, "condition": "Rain"},
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _bootstrap_mean_ci(values, seed=42, repetitions=2000):
    values = [float(value) for value in values]
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _signature(candidate):
    return tuple(
        sorted(
            (
                item["slot"],
                item.get("type"),
                item.get("colour"),
            )
            for item in candidate["items"]
        )
    )


def _catalogue_summary(items, *, abstract):
    by_slot = Counter(item["slot"] for item in items)
    by_colour = Counter(item.get("colour") for item in items if item.get("colour"))
    by_style = Counter(
        style.strip()
        for item in items
        for style in str(item.get("style", "")).split(";")
        if style.strip()
    )
    by_type = Counter(
        f'{item["slot"]}|{broad_garment_category(item["slot"], item.get("type", ""))}'
        for item in items
    )
    support_by_colour = Counter()
    for item in items:
        if item.get("colour"):
            support_by_colour[item["colour"]] += int(item.get("support_count", 1))
    visible_states = {
        (
            item["slot"],
            broad_garment_category(item["slot"], item.get("type", "")),
            item.get("colour"),
        )
        for item in items
    }
    support = [int(item.get("support_count", 1)) for item in items]
    forbidden = {"brand", "price", "source_url", "title"}
    return {
        "records": len(items),
        "records_by_slot": dict(sorted(by_slot.items())),
        "visible_type_colour_states": len(visible_states),
        "distinct_types": len(
            {
                (item["slot"], broad_garment_category(item["slot"], item.get("type", "")))
                for item in items
            }
        ),
        "distinct_colours": len({item.get("colour") for item in items if item.get("colour")}),
        "distinct_styles": len(
            {
                style.strip()
                for item in items
                for style in str(item.get("style", "")).split(";")
                if style.strip()
            }
        ),
        "distribution": {
            "by_colour": dict(sorted(by_colour.items())),
            "by_style": dict(sorted(by_style.items())),
            "by_slot_and_type": dict(sorted(by_type.items())),
            "underlying_support_by_colour": dict(sorted(support_by_colour.items())),
            "largest_record_colour_share": max(by_colour.values()) / len(items),
            "largest_record_style_share": max(by_style.values()) / len(items),
            "largest_slot_share": max(by_slot.values()) / len(items),
            "singleton_support_prototypes": sum(value == 1 for value in support),
        },
        "support_count": _summary(support),
        "privacy_contract": {
            "all_abstract": all(item.get("abstract") for item in items) if abstract else False,
            "nonempty_image_paths": sum(bool(item.get("image_path")) for item in items),
            "records_with_forbidden_commercial_fields": sum(
                bool(forbidden & set(item)) for item in items
            ),
        },
    }


def _positive_query_pools(dataset_path):
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    pools = defaultdict(list)
    for example_index in torch.where(payload["labels"].eq(1))[0].tolist():
        for slot_index, slot in enumerate(SLOTS):
            if bool(payload["masks"][example_index, slot_index]):
                pools[slot].append(payload["embeddings"][example_index, slot_index].float())
    return pools


def _items_by_slot(items):
    return {slot: [item for item in items if item["slot"] == slot] for slot in SLOTS}


def _attribute_changes(primary, alternative):
    primary_values = {
        item["slot"]: (item.get("type"), item.get("colour"))
        for item in primary["items"]
    }
    return sum(
        primary_values.get(item["slot"]) != (item.get("type"), item.get("colour"))
        for item in alternative["items"]
        if item["slot"] in primary_values
    )


def _weather_coverage(items):
    rows = []
    for scenario, weather in WEATHER_SCENARIOS.items():
        filtered, constraints = _weather_filter(items, weather)
        for style in STYLES:
            styled = _style_filter(filtered, style)
            counts = Counter(item["slot"] for item in styled)
            if weather["feels_like"] >= 28:
                counts["outer_top"] = 1  # runtime masked No Outer Layer state
            rows.append(
                {
                    "scenario": scenario,
                    "style": style,
                    "constraints": constraints,
                    "counts_by_slot": {slot: counts[slot] for slot in SLOTS},
                    "all_slots_available": all(counts[slot] > 0 for slot in SLOTS),
                    "uses_masked_no_outer_state": weather["feels_like"] >= 28,
                }
            )
    return {
        "cases": len(rows),
        "all_cases_have_four_logical_slots": all(row["all_slots_available"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--abstract", default="models/polyvore_abstract_prototypes.pt")
    parser.add_argument("--legacy-catalogue", default="models/catalogue_embeddings.pt")
    parser.add_argument("--queries", default="data/processed_disjoint/test.pt")
    parser.add_argument("--query-count", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", default="results/v3_system_comparison.json")
    parser.add_argument("--device")
    parser.add_argument("--audience", default="womenswear")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    abstract_payload = torch.load(args.abstract, map_location="cpu", weights_only=False)
    legacy_payload = torch.load(args.legacy_catalogue, map_location="cpu", weights_only=False)
    abstract_items = abstract_payload["items"]
    selected_abstract_items = _audience_filter(abstract_items, args.audience)
    legacy_items = legacy_payload["items"]
    query_pools = _positive_query_pools(args.queries)
    ranker = OutfitCandidateRanker(checkpoint_path, device=args.device)

    rows = []
    for query_index in range(args.query_count):
        style = STYLES[query_index % len(STYLES)]
        input_slot = SLOTS[query_index % len(SLOTS)]
        pool = query_pools[input_slot]
        input_embedding = pool[(query_index // len(SLOTS)) % len(pool)]
        fixed_item = {
            "item_id": f"held-out-query-{query_index}",
            "slot": input_slot,
            "type": input_slot,
            "colour": INPUT_COLOURS[query_index % len(INPUT_COLOURS)],
            "embedding": input_embedding,
        }

        abstract_styled = _style_filter(selected_abstract_items, style)
        full_by_slot = _items_by_slot(abstract_styled)
        full_candidates = {
            slot: values for slot, values in full_by_slot.items() if slot != input_slot
        }
        exact_started = time.perf_counter()
        exact = ranker.rank_exact(
            {input_slot: fixed_item},
            full_candidates,
            top_k=256,
            batch_size=args.batch_size,
            maximum_combinations=5_000_000,
        )
        exact_elapsed = time.perf_counter() - exact_started
        exact_stats = dict(ranker.last_search_stats)

        short_by_slot = {
            slot: _shortlist_candidates(values, input_embedding, limit=10)
            for slot, values in full_by_slot.items()
            if slot != input_slot
        }
        quota_started = time.perf_counter()
        quota = ranker.rank({input_slot: fixed_item}, short_by_slot, limit=1000)
        quota_elapsed = time.perf_counter() - quota_started

        legacy_styled = _style_filter(legacy_items, style)
        legacy_by_slot = _items_by_slot(legacy_styled)
        legacy_short = {
            slot: _shortlist_candidates(values, input_embedding, limit=10)
            for slot, values in legacy_by_slot.items()
            if slot != input_slot
        }
        legacy_started = time.perf_counter()
        legacy_ranked = ranker.rank(
            {input_slot: fixed_item}, legacy_short, limit=1000
        )
        legacy_elapsed = time.perf_counter() - legacy_started

        recommendation_slots = [slot for slot in SLOTS if slot != input_slot]
        rule_started = time.perf_counter()
        rule_ranked = RULE_RANKER.rank(
            INPUT_COLOURS[query_index % len(INPUT_COLOURS)],
            style,
            recommendation_slots,
            None,
        )
        rule_elapsed = time.perf_counter() - rule_started

        exact_signatures = [_signature(candidate) for candidate in exact]
        exact_top10 = set(exact_signatures[:10])
        quota_signatures = [_signature(candidate) for candidate in quota]
        quota_primary, quota_alternative = choose_diverse_pair(quota)
        exact_primary, exact_alternative = choose_diverse_pair(exact)
        rows.append(
            {
                "query": query_index,
                "held_out_split": str(args.queries),
                "style": style,
                "input_slot": input_slot,
                "v3_exact": {
                    "best_score": exact[0]["compatibility_score"],
                    "elapsed_seconds": exact_elapsed,
                    "combinations": exact_stats["combinations_evaluated"],
                    "top_k_unique_visible": len(set(exact_signatures)),
                    "primary_alternative_attribute_changes": _attribute_changes(
                        exact_primary, exact_alternative
                    ),
                },
                "quota_ablation_same_abstract_space": {
                    "best_score": quota[0]["compatibility_score"],
                    "elapsed_seconds": quota_elapsed,
                    "combinations": len(quota),
                    "search_space_fraction": len(quota)
                    / exact_stats["combinations_evaluated"],
                    "global_top1_recovered": quota_signatures[0]
                    == exact_signatures[0],
                    "global_top1_rank_if_within_exact_top256": (
                        exact_signatures.index(quota_signatures[0]) + 1
                        if quota_signatures[0] in exact_signatures
                        else None
                    ),
                    "top10_recall": len(set(quota_signatures[:10]) & exact_top10) / 10,
                    "model_score_regret": max(
                        0.0,
                        exact[0]["compatibility_score"]
                        - quota[0]["compatibility_score"],
                    ),
                    "primary_alternative_attribute_changes": _attribute_changes(
                        quota_primary, quota_alternative
                    ),
                },
                "v2_zara_quota_runtime": {
                    "elapsed_seconds": legacy_elapsed,
                    "combinations": len(legacy_ranked),
                    "best_score_not_cross_catalogue_comparable": legacy_ranked[0][
                        "compatibility_score"
                    ],
                },
                "v1_rule_runtime": {
                    "elapsed_seconds": rule_elapsed,
                    "combinations": len(rule_ranked),
                    "best_rule_score_percent": rule_ranked[0][0],
                },
            }
        )

    # One real repeat checks deterministic ordering in addition to unit tests.
    first = rows[0]
    style = first["style"]
    input_slot = first["input_slot"]
    input_embedding = query_pools[input_slot][0]
    fixed_item = {
        "item_id": "determinism-query",
        "slot": input_slot,
        "type": input_slot,
        "colour": "Black",
        "embedding": input_embedding,
    }
    repeat_candidates = {
        slot: values
        for slot, values in _items_by_slot(_style_filter(selected_abstract_items, style)).items()
        if slot != input_slot
    }
    repeat_a = ranker.rank_exact(
        {input_slot: fixed_item}, repeat_candidates, top_k=10, batch_size=args.batch_size,
        maximum_combinations=5_000_000,
    )
    repeat_b = ranker.rank_exact(
        {input_slot: fixed_item}, repeat_candidates, top_k=10, batch_size=args.batch_size,
        maximum_combinations=5_000_000,
    )
    deterministic = (
        [_signature(item) for item in repeat_a] == [_signature(item) for item in repeat_b]
        and [item["compatibility_score"] for item in repeat_a]
        == [item["compatibility_score"] for item in repeat_b]
    )

    regrets = [row["quota_ablation_same_abstract_space"]["model_score_regret"] for row in rows]
    exact_latencies = [row["v3_exact"]["elapsed_seconds"] for row in rows]
    quota_latencies = [
        row["quota_ablation_same_abstract_space"]["elapsed_seconds"] for row in rows
    ]
    v2_latencies = [row["v2_zara_quota_runtime"]["elapsed_seconds"] for row in rows]
    v1_latencies = [row["v1_rule_runtime"]["elapsed_seconds"] for row in rows]
    compatibility = json.loads(
        Path("results/compatibility_test_metrics.json").read_text(encoding="utf-8")
    )
    bias = json.loads(Path("results/disjoint_bias_audit.json").read_text(encoding="utf-8"))
    disjoint = json.loads(
        Path("results/disjoint_release_check.json").read_text(encoding="utf-8")
    )

    report = {
        "version_definitions": {
            "v1": "Interpretable rule/content ranker over authored colour palettes.",
            "v2": "Deployed compatibility head with a 300-item Zara candidate catalogue and per-slot quota shortlist.",
            "v3": "D2 disjoint compatibility head with user-selected, brand-free Polyvore-derived abstract states and batched exact global top-k search.",
            "d2_promotion": "The disjoint head passed after correcting shuffled FITB answer labels by outfit set_id.",
        },
        "fair_comparison_boundaries": {
            "same_checkpoint_v2_v3": True,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "consequence": "V2 and V3 have identical outfit-classification AUC/AP/accuracy. Search metrics, candidate coverage and latency are the valid V3 comparison axes.",
            "score_warning": "Model-score improvement measures optimisation of the deployed head, not human preference or purchase accuracy.",
            "cross_catalogue_warning": "Absolute scores from Zara and abstract prototype catalogues are not interpreted as a quality comparison because candidate distributions differ.",
        },
        "shared_head_metrics_v2_and_v3": compatibility["metrics"],
        "shared_head_checkpoint_metadata": compatibility["checkpoint_metadata"],
        "three_seed_operational_bias_metrics": bias["mitigation_result"]["summary"],
        "candidate_spaces": {
            "v2_zara": _catalogue_summary(legacy_items, abstract=False),
            "v3_abstract": _catalogue_summary(abstract_items, abstract=True),
            "v3_selected_audience": {
                "audience": args.audience,
                **_catalogue_summary(selected_abstract_items, abstract=True),
            },
        },
        "search_ablation_same_abstract_candidate_space": {
            "queries": len(rows),
            "query_source": str(args.queries),
            "query_policy": "Positive held-out disjoint outfits; deterministic slot/style rotation. Ground-truth items are not used as candidate targets.",
            "global_top1_recovery_rate_quota": statistics.mean(
                row["quota_ablation_same_abstract_space"]["global_top1_recovered"]
                for row in rows
            ),
            "mean_top10_recall_quota": statistics.mean(
                row["quota_ablation_same_abstract_space"]["top10_recall"]
                for row in rows
            ),
            "model_score_regret_quota": {
                **_summary(regrets),
                "bootstrap_95_percent_ci_for_mean": _bootstrap_mean_ci(regrets),
                "queries_with_positive_regret": sum(value > 1e-12 for value in regrets),
            },
            "mean_search_space_fraction_examined_by_quota": statistics.mean(
                row["quota_ablation_same_abstract_space"]["search_space_fraction"]
                for row in rows
            ),
            "deterministic_repeat_top10": deterministic,
        },
        "runtime_seconds_same_machine": {
            "v1_rule": _summary(v1_latencies),
            "v2_zara_quota": _summary(v2_latencies),
            "v3_quota_ablation_on_abstract": _summary(quota_latencies),
            "v3_exact": _summary(exact_latencies),
            "device": str(ranker.device),
            "batch_size": args.batch_size,
        },
        "weather_style_coverage": _weather_coverage(selected_abstract_items),
        "research_candidate_gate": {
            "promotion_allowed": disjoint["production_promotion_allowed"],
            "decision": disjoint["decision"],
            "blocking_failures": disjoint["blocking_failures"],
        },
        "per_query": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
