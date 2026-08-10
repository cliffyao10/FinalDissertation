"""Small neural scorer trained on frozen SigLIP item embeddings."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    embedding_dim: int = 768
    hidden_dim: int = 256
    slot_dim: int = 32
    number_of_slots: int = 4
    dropout: float = 0.20


class CompatibilityRanker(nn.Module):
    """Return one compatibility logit for a masked four-slot outfit."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.image_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.slot_embedding = nn.Embedding(config.number_of_slots, config.slot_dim)
        item_dim = config.hidden_dim + config.slot_dim
        self.scorer = nn.Sequential(
            nn.Linear(item_dim * 2, config.hidden_dim),
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
        slot_ids = torch.arange(slot_count, device=embeddings.device)
        slot_ids = slot_ids.unsqueeze(0).expand(batch_size, -1)
        projected = self.image_projection(embeddings)
        items = torch.cat((projected, self.slot_embedding(slot_ids)), dim=-1)

        float_mask = mask.float().unsqueeze(-1)
        pooled = (items * float_mask).sum(dim=1)
        pooled = pooled / float_mask.sum(dim=1).clamp_min(1.0)

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

    def probability(self, embeddings, mask):
        return torch.sigmoid(self.forward(embeddings, mask))


def save_checkpoint(path, model, extra=None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CompatibilityRanker(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint.get("extra", {})
