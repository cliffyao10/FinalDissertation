"""Shared frozen SigLIP image-embedding helpers."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL = "google/siglip-base-patch16-224"


def pooled_image_features(output):
    """Normalise the Transformers 4.x tensor and 5.x model-output APIs."""

    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError("SigLIP image encoder did not return pooled features.")
    return pooled


def load_encoder(model_name=DEFAULT_MODEL, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    # Feature extraction is deliberately separate from compatibility training.
    # Make the frozen state explicit as well as disabling autograd in embed_paths.
    model.requires_grad_(False)
    model.eval()
    return processor, model, device


@torch.inference_mode()
def embed_paths(paths, processor, model, device):
    images = []
    for path in paths:
        with Image.open(Path(path)) as image:
            images.append(image.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    embeddings = pooled_image_features(model.get_image_features(**inputs))
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.detach().cpu()
