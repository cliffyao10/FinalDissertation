"""
Clothing category recognition module.

This module uses a pretrained CLIP model to perform
zero-shot clothing category classification.
"""

from functools import lru_cache

from PIL import Image
from transformers import pipeline


MODEL_NAME = "openai/clip-vit-base-patch32"

CLOTHING_CATEGORIES = [
    "T-shirt",
    "shirt",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "jeans",
    "trousers",
    "shorts",
    "skirt",
    "dress",
    "shoes",
]


@lru_cache(maxsize=1)
def load_category_model():
    """
    Load the pretrained CLIP model once and reuse it.

    Returns:
        Pipeline: Zero-shot image classification pipeline.
    """

    return pipeline(
        task="zero-shot-image-classification",
        model=MODEL_NAME,
    )


def predict_category_with_confidence(image):
    """
    Predict the clothing category and confidence score.

    Parameters:
        image (PIL.Image.Image): Uploaded clothing image.

    Returns:
        dict: Predicted category, confidence and method.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("The image must be a PIL Image.")

    classifier = load_category_model()

    predictions = classifier(
        image.convert("RGB"),
        candidate_labels=CLOTHING_CATEGORIES,
    )

    best_prediction = predictions[0]

    return {
        "category": best_prediction["label"].title(),
        "confidence": float(best_prediction["score"]),
        "method": "CLIP zero-shot",
    }


def predict_category(image):
    """
    Return only the predicted category.

    This keeps the function compatible with the current Streamlit app.
    """

    result = predict_category_with_confidence(image)
    return result["category"]