"""Fast structural test that makes no model-quality claim."""

import tempfile
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommendation_training.compatibility_model import (
    CompatibilityRanker,
    ModelConfig,
    load_checkpoint,
    save_checkpoint,
)


def main():
    model = CompatibilityRanker(ModelConfig(embedding_dim=16, hidden_dim=32, slot_dim=8))
    model.eval()
    embeddings = torch.randn(5, 4, 16)
    mask = torch.ones(5, 4, dtype=torch.bool)
    output = model(embeddings, mask)
    assert output.shape == (5,)
    assert torch.isfinite(output).all()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(path, model, {"test": True})
        restored, metadata = load_checkpoint(path)
        assert metadata["test"] is True
        assert torch.allclose(model(embeddings, mask), restored(embeddings, mask))
    print("Compatibility model smoke test passed.")


if __name__ == "__main__":
    main()
