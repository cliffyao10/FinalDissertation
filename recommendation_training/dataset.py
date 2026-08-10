"""Dataset utilities for pre-computed four-slot outfit embeddings."""

import torch
from torch.utils.data import Dataset


SLOTS = ("inner_top", "outer_top", "bottom", "shoes")


class OutfitDataset(Dataset):
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.embeddings = payload["embeddings"].float()
        self.masks = payload["masks"].bool()
        self.labels = payload["labels"].float()
        if self.embeddings.ndim != 3 or self.embeddings.shape[1] != len(SLOTS):
            raise ValueError("Expected embeddings shaped [examples, 4, dimension].")
        if self.masks.shape != self.embeddings.shape[:2]:
            raise ValueError("Masks must have shape [examples, 4].")
        if len(self.labels) != len(self.embeddings):
            raise ValueError("Labels and embeddings have different lengths.")
        if not bool(self.masks.any(dim=1).all()):
            raise ValueError("Every example needs at least one present item.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "embeddings": self.embeddings[index],
            "mask": self.masks[index],
            "label": self.labels[index],
        }
