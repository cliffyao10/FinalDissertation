"""Rank complete catalogue outfits with a trained checkpoint."""

from itertools import product

import torch

from .compatibility_model import load_checkpoint


SLOTS = ("inner_top", "outer_top", "bottom", "shoes")


class OutfitCandidateRanker:
    def __init__(self, checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata = load_checkpoint(checkpoint, self.device)

    @torch.no_grad()
    def rank(self, fixed_items, candidates_by_slot, limit=1000):
        if limit <= 0:
            raise ValueError("Ranking limit must be positive.")
        choices = []
        for slot in SLOTS:
            choices.append(
                [fixed_items[slot]]
                if slot in fixed_items
                else candidates_by_slot.get(slot, [])
            )
        if any(not slot_choices for slot_choices in choices):
            raise ValueError("Every unfilled slot needs at least one candidate.")

        combinations = []
        for index, outfit in enumerate(product(*choices)):
            if index >= limit:
                break
            combinations.append(outfit)
        if not combinations:
            raise ValueError("No complete outfit combinations were generated.")
        embeddings = torch.stack(
            [torch.stack([item["embedding"] for item in outfit]) for outfit in combinations]
        ).to(self.device)
        if embeddings.shape[-1] != self.model.config.embedding_dim:
            raise ValueError(
                "Catalogue embedding dimension does not match the trained checkpoint "
                f"({embeddings.shape[-1]} != {self.model.config.embedding_dim})."
            )
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=self.device)
        probabilities = self.model.probability(embeddings, mask).cpu().tolist()
        ranked = sorted(
            zip(probabilities, combinations), key=lambda result: result[0], reverse=True
        )
        return [
            {"compatibility_score": score, "items": list(items)}
            for score, items in ranked
        ]


def choose_diverse_pair(ranked, minimum_colour_changes=2, minimum_score_ratio=0.80):
    if not ranked:
        return None, None
    primary = ranked[0]
    primary_colours = [item.get("colour") for item in primary["items"]]
    minimum_score = primary["compatibility_score"] * minimum_score_ratio
    alternative = next(
        (
            candidate
            for candidate in ranked[1:]
            if candidate["compatibility_score"] >= minimum_score
            and sum(
                left != right
                for left, right in zip(
                    primary_colours,
                    [item.get("colour") for item in candidate["items"]],
                )
            ) >= minimum_colour_changes
        ),
        ranked[1] if len(ranked) > 1 else None,
    )
    return primary, alternative
