"""Rank complete catalogue outfits with a trained checkpoint."""

import heapq
import time

import torch

from .compatibility_model import load_checkpoint


SLOTS = ("inner_top", "outer_top", "bottom", "shoes")


def item_occupied_slots(item):
    """Return an item's occupied slots, accepting CSV or in-memory values."""

    if item is None:
        return set()
    value = item.get("occupies_slots") or item.get("slot")
    if isinstance(value, (list, tuple, set)):
        occupied = {str(slot) for slot in value if slot}
    else:
        occupied = {
            slot.strip()
            for slot in str(value).replace(",", ";").split(";")
            if slot.strip()
        }
    invalid = occupied - set(SLOTS)
    if invalid:
        raise ValueError(f"Unsupported occupied slots: {sorted(invalid)}")
    return occupied


def item_masked_from_model(item):
    """Return whether a visible conceptual state represents an absent item."""

    return bool(item is not None and item.get("masked_from_model"))


class OutfitCandidateRanker:
    def __init__(self, checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata = load_checkpoint(checkpoint, self.device)
        self.last_search_stats = {}

    def _iter_combinations(self, fixed_items, candidates_by_slot):
        """Yield every valid complete outfit without materialising the product."""

        fixed_slots = set(fixed_items)

        def build(slot_index, covered, outfit):
            if slot_index == len(SLOTS):
                if covered == set(SLOTS):
                    yield tuple(outfit)
                return
            slot = SLOTS[slot_index]
            if slot in covered:
                if slot in fixed_items:
                    return
                yield from build(slot_index + 1, covered, [*outfit, None])
                return
            choices = (
                [fixed_items[slot]]
                if slot in fixed_items
                else candidates_by_slot.get(slot, [])
            )
            for item in choices:
                occupied = item_occupied_slots(item)
                conflicts_with_fixed = (
                    slot not in fixed_items
                    and bool((occupied - {slot}) & fixed_slots)
                )
                if (
                    slot not in occupied
                    or occupied & covered
                    or conflicts_with_fixed
                ):
                    continue
                yield from build(
                    slot_index + 1,
                    covered | occupied,
                    [*outfit, item],
                )

        yield from build(0, set(), [])

    @torch.no_grad()
    def rank_exact(
        self,
        fixed_items,
        candidates_by_slot,
        top_k=256,
        batch_size=4096,
        maximum_combinations=None,
    ):
        """Return the global top-k after streaming every valid combination.

        ``maximum_combinations`` is a safety gate, not a prefix cutoff.  The
        method raises rather than returning when the gate is exceeded, so a
        partial search can never be mistaken for an exact result.
        """

        if top_k <= 0 or batch_size <= 0:
            raise ValueError("top_k and batch_size must be positive.")

        started = time.perf_counter()
        unique_items = []
        item_indexes = {}
        for item in list(fixed_items.values()) + [
            item
            for slot_items in candidates_by_slot.values()
            for item in slot_items
        ]:
            if item_masked_from_model(item):
                continue
            identity = id(item)
            if identity not in item_indexes:
                item_indexes[identity] = len(unique_items)
                unique_items.append(item)
        if not unique_items:
            raise ValueError("No items were supplied for outfit search.")

        dimension = torch.as_tensor(unique_items[0]["embedding"]).numel()
        embeddings = torch.stack(
            [torch.as_tensor(item["embedding"]).float().flatten() for item in unique_items]
        ).to(self.device)
        if embeddings.shape[-1] != self.model.config.embedding_dim:
            raise ValueError(
                "Catalogue embedding dimension does not match the trained checkpoint "
                f"({embeddings.shape[-1]} != {self.model.config.embedding_dim})."
            )
        item_slots = []
        for item in unique_items:
            occupied = item_occupied_slots(item)
            anchor = next((slot for slot in SLOTS if slot in occupied), None)
            if anchor is None:
                raise ValueError("Every candidate must occupy a supported slot.")
            item_slots.append(SLOTS.index(anchor))
        slot_ids = torch.tensor(item_slots, dtype=torch.long, device=self.device)
        projected = self.model.image_projection(embeddings)
        cached_features = torch.cat(
            (projected, self.model.slot_embedding(slot_ids)), dim=-1
        )
        zero_feature = torch.zeros(
            1, cached_features.shape[-1], device=self.device
        )
        feature_table = torch.cat((cached_features, zero_feature), dim=0)
        missing_index = len(unique_items)

        heap = []
        evaluated = 0
        sequence = 0
        pending = []

        def score_pending():
            nonlocal evaluated, sequence
            if not pending:
                return
            indexes = torch.tensor(
                [
                    [
                        item_indexes[id(item)]
                        if item is not None and not item_masked_from_model(item)
                        else missing_index
                        for item in outfit
                    ]
                    for outfit in pending
                ],
                dtype=torch.long,
                device=self.device,
            )
            mask = indexes.ne(missing_index)
            features = feature_table[indexes]
            logits = self.model.score_preprojected(features, mask)
            temperature = max(float(self.model.calibration_temperature), 1e-6)
            probabilities = torch.sigmoid(logits / temperature).cpu().tolist()
            for score, outfit in zip(probabilities, pending):
                entry = (float(score), sequence, outfit)
                sequence += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
            evaluated += len(pending)
            pending.clear()

        for combination in self._iter_combinations(fixed_items, candidates_by_slot):
            if maximum_combinations is not None and evaluated + len(pending) >= maximum_combinations:
                raise RuntimeError(
                    "Exact-search safety limit exceeded; no partial ranking was returned."
                )
            pending.append(combination)
            if len(pending) >= batch_size:
                score_pending()
        score_pending()
        if not evaluated:
            raise ValueError("No complete outfit combinations were generated.")

        ranked_entries = sorted(heap, key=lambda entry: (-entry[0], entry[1]))
        elapsed = time.perf_counter() - started
        self.last_search_stats = {
            "method": "batched_exact_global_top_k",
            "exact": True,
            "combinations_evaluated": evaluated,
            "top_k_retained": len(ranked_entries),
            "batch_size": batch_size,
            "device": str(self.device),
            "elapsed_seconds": elapsed,
            "combinations_per_second": evaluated / max(elapsed, 1e-9),
            "unique_item_projections": len(unique_items),
        }
        return [
            {
                "compatibility_score": score,
                "items": [item for item in items if item is not None],
            }
            for score, _, items in ranked_entries
        ]

    @torch.no_grad()
    def rank(self, fixed_items, candidates_by_slot, limit=1000):
        if limit <= 0:
            raise ValueError("Ranking limit must be positive.")
        combinations = []
        fixed_slots = set(fixed_items)
        def build(slot_index, covered, outfit):
            if len(combinations) >= limit:
                return
            if slot_index == len(SLOTS):
                if covered == set(SLOTS):
                    combinations.append(tuple(outfit))
                return
            slot = SLOTS[slot_index]
            if slot in covered:
                if slot in fixed_items:
                    return
                build(slot_index + 1, covered, [*outfit, None])
                return
            choices = (
                [fixed_items[slot]]
                if slot in fixed_items
                else candidates_by_slot.get(slot, [])
            )
            for item in choices:
                occupied = item_occupied_slots(item)
                conflicts_with_fixed = (
                    slot not in fixed_items
                    and bool((occupied - {slot}) & fixed_slots)
                )
                if (
                    slot not in occupied
                    or occupied & covered
                    or conflicts_with_fixed
                ):
                    continue
                build(slot_index + 1, covered | occupied, [*outfit, item])

        build(0, set(), [])
        if not combinations:
            raise ValueError("No complete outfit combinations were generated.")
        dimension = next(
            torch.as_tensor(item["embedding"]).numel()
            for outfit in combinations
            for item in outfit
            if item is not None and not item_masked_from_model(item)
        )
        embeddings = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(item["embedding"]).flatten()
                        if item is not None and not item_masked_from_model(item)
                        else torch.zeros(dimension)
                        for item in outfit
                    ]
                )
                for outfit in combinations
            ]
        ).to(self.device)
        if embeddings.shape[-1] != self.model.config.embedding_dim:
            raise ValueError(
                "Catalogue embedding dimension does not match the trained checkpoint "
                f"({embeddings.shape[-1]} != {self.model.config.embedding_dim})."
            )
        mask = torch.tensor(
            [
                [
                    item is not None and not item_masked_from_model(item)
                    for item in outfit
                ]
                for outfit in combinations
            ],
            dtype=torch.bool,
            device=self.device,
        )
        probabilities = self.model.probability(embeddings, mask).cpu().tolist()
        ranked = sorted(
            zip(probabilities, combinations), key=lambda result: result[0], reverse=True
        )
        return [
            {
                "compatibility_score": score,
                "items": [item for item in items if item is not None],
            }
            for score, items in ranked
        ]


def choose_diverse_pair(ranked, minimum_colour_changes=2, minimum_score_ratio=0.80):
    if not ranked:
        return None, None
    primary = ranked[0]
    primary_attributes = {
        item["slot"]: (item.get("type"), item.get("colour"))
        for item in primary["items"]
    }
    minimum_score = primary["compatibility_score"] * minimum_score_ratio
    alternative = next(
        (
            candidate
            for candidate in ranked[1:]
            if candidate["compatibility_score"] >= minimum_score
            and sum(
                primary_attributes.get(item["slot"])
                != (item.get("type"), item.get("colour"))
                for item in candidate["items"]
                if item["slot"] in primary_attributes
            ) >= minimum_colour_changes
        ),
        ranked[1] if len(ranked) > 1 else None,
    )
    return primary, alternative
