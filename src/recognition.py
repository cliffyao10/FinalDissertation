"""SigLIP-based clothing recognition.

The module predicts three kinds of information:
1. A fine garment category and its broader parent category.
2. A semantic garment colour.
3. Multiple broad style cues for recommendation.

The hierarchical category output allows the recommendation system to keep
working when visually similar fine categories are uncertain.
"""

from functools import lru_cache

import torch
from transformers import pipeline

from recommendation_training.embedding import pooled_image_features


MODEL_NAME = "google/siglip-base-patch16-224"


# Fine categories are intentionally limited to visually meaningful classes.
# Multiple prompts may map to the same output label.
CATEGORY_PROMPTS = {
    "a photo of a fitted camisole or tank top with narrow shoulder straps": (
        "Tank Top",
        "Top",
    ),
    "a photo of a sleeveless casual tank top": (
        "Tank Top",
        "Top",
    ),
    "a photo of a short-sleeved crew-neck T-shirt": (
        "T-Shirt",
        "Top",
    ),
    "a photo of a collared button-up shirt": (
        "Shirt",
        "Top",
    ),
    "a photo of a women's blouse": (
        "Blouse",
        "Top",
    ),
    "a photo of a knitted sweater or pullover": (
        "Sweater",
        "Top",
    ),
    "a photo of a soft cotton hooded sweatshirt": (
        "Hoodie",
        "Top",
    ),
    (
        "a photo of an outdoor jacket, windbreaker, waterproof shell, "
        "or technical jacket"
    ): (
        "Jacket",
        "Outerwear",
    ),
    "a photo of a short hip-length zipped technical shell jacket": (
        "Jacket",
        "Outerwear",
    ),
    "a photo of a structured blazer or suit jacket": (
        "Blazer",
        "Outerwear",
    ),
    "a photo of a long heavy outer coat extending below the hips or knees": (
        "Coat",
        "Outerwear",
    ),
    "a photo of denim jeans": (
        "Jeans",
        "Bottom",
    ),
    "a photo of tailored trousers or casual pants": (
        "Trousers",
        "Bottom",
    ),
    "a photo of shorts": (
        "Shorts",
        "Bottom",
    ),
    "a photo of a skirt": (
        "Skirt",
        "Bottom",
    ),
    "a photo of a one-piece dress": (
        "Dress",
        "One-piece",
    ),
    "a photo of a two-piece bikini swimsuit": (
        "Bikini",
        "Swimwear",
    ),
    "a photo of a one-piece swimsuit or swimming costume": (
        "Swimsuit",
        "Swimwear",
    ),
    "a photo of footwear or shoes": (
        "Shoes",
        "Footwear",
    ),
}


COLOUR_PROMPTS = {
    f"a piece of {colour.lower()} clothing": colour
    for colour in (
        "Black",
        "White",
        "Grey",
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Purple",
        "Pink",
        "Brown",
        "Beige",
    )
}


# These are broad recommendation-oriented style cues. They are multi-label:
# an item may be both Outdoor and Sporty, or Casual and Minimalist.
STYLE_PROMPTS = {
    "casual everyday clothing with a relaxed appearance": "Casual",
    "formal clothing for ceremonies or elegant occasions": "Formal",
    "business or professional office clothing": "Business",
    "sporty athletic or activewear clothing": "Sporty",
    "outdoor technical hiking or weather-protective clothing": "Outdoor",
    "urban streetwear clothing": "Streetwear",
    "minimalist clothing with a simple clean design": "Minimalist",
    "party or evening social-event clothing": "Party",
    "beachwear, swimwear, or clothing for swimming": "Beachwear",
}


# The current rule baseline has fewer categories. These aliases preserve the
# fine recognition result while giving the baseline a compatible input.
RECOMMENDATION_CATEGORY_ALIASES = {
    "Tank Top": "T-Shirt",
    "Blouse": "Shirt",
    "Blazer": "Jacket",
}

PARENT_RECOMMENDATION_ALIASES = {
    "Top": "T-Shirt",
    "Outerwear": "Jacket",
    "Bottom": "Trousers",
    "One-piece": "Dress",
    "Footwear": "Shoes",
    "Swimwear": "Swimwear",
}


@lru_cache(maxsize=1)
def load_siglip_classifier():
    """Load SigLIP once and reuse it for all recognition tasks."""

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="zero-shot-image-classification",
        model=MODEL_NAME,
        device=device,
    )


def extract_image_embedding(image):
    """Return the frozen SigLIP image vector used by recommendation training."""

    classifier = load_siglip_classifier()
    processor = classifier.image_processor
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    model_device = classifier.model.device
    inputs = {name: value.to(model_device) for name, value in inputs.items()}
    with torch.inference_mode():
        embedding = pooled_image_features(
            classifier.model.get_image_features(**inputs)
        )
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding[0].detach().cpu()


def _rank_prompt_predictions(image, candidate_prompts):
    classifier = load_siglip_classifier()

    predictions = classifier(
        image.convert("RGB"),
        candidate_labels=list(candidate_prompts),
    )

    return sorted(
        (
            (prediction["label"], float(prediction["score"]))
            for prediction in predictions
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def _aggregate_labels(ranked_prompts, prompt_to_label):
    """Use the strongest prompt score for every output label."""

    label_scores = {}

    for prompt, score in ranked_prompts:
        label = prompt_to_label[prompt]
        label_scores[label] = max(
            score,
            label_scores.get(label, 0.0),
        )

    return sorted(
        label_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def _prediction_summary(ranked_labels):
    best_label, best_score = ranked_labels[0]
    second_score = ranked_labels[1][1] if len(ranked_labels) > 1 else 0.0

    relative_margin = (
        (best_score - second_score) / best_score
        if best_score > 0
        else 0.0
    )

    return {
        "label": best_label,
        "confidence": best_score,
        "margin": best_score - second_score,
        "relative_margin": relative_margin,
        "score_distribution": dict(ranked_labels),
        "model": MODEL_NAME,
    }


def predict_category_with_confidence(image):
    """Predict both a fine category and a robust parent category."""

    prompt_to_fine = {
        prompt: category_data[0]
        for prompt, category_data in CATEGORY_PROMPTS.items()
    }

    ranked_prompts = _rank_prompt_predictions(
        image,
        CATEGORY_PROMPTS,
    )

    ranked_fine = _aggregate_labels(
        ranked_prompts,
        prompt_to_fine,
    )

    fine_summary = _prediction_summary(ranked_fine)
    fine_category = fine_summary["label"]

    # SigLIP scores are similarities rather than calibrated probabilities.
    # This conservative threshold only catches extremely weak evidence where
    # every supplied category prompt is a poor visual match.
    category_out_of_scope = fine_summary["confidence"] < 0.001

    fine_to_parent = {
        fine: parent
        for fine, parent in CATEGORY_PROMPTS.values()
    }

    parent_scores = {}

    for fine, score in ranked_fine:
        parent = fine_to_parent[fine]
        parent_scores[parent] = max(
            score,
            parent_scores.get(parent, 0.0),
        )

    ranked_parents = sorted(
        parent_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    parent_summary = _prediction_summary(ranked_parents)
    fine_category_confident = (
        fine_summary["relative_margin"] >= 0.20
        and not category_out_of_scope
    )

    # Recommendation intentionally uses the broad garment group. Fine labels
    # remain available for evaluation, but Jacket versus Coat should not alter
    # the user flow or create unnecessary recommendation branches.
    recommendation_category = PARENT_RECOMMENDATION_ALIASES[
        parent_summary["label"]
    ]

    likely_fine_categories = [
        candidate_category
        for candidate_category, _ in ranked_fine
        if fine_to_parent[candidate_category] == parent_summary["label"]
    ][:3]

    if category_out_of_scope:
        displayed_category = "Unknown"
    elif fine_category_confident:
        displayed_category = fine_category
    else:
        displayed_category = parent_summary["label"]

    displayed_parent = (
        "Unknown"
        if category_out_of_scope
        else parent_summary["label"]
    )

    if category_out_of_scope:
        recommendation_category = "Unknown"

    return {
        "category": displayed_category,
        "predicted_fine_category": fine_category,
        "likely_fine_categories": likely_fine_categories,
        "parent_category": displayed_parent,
        "recommendation_category": recommendation_category,
        "confidence": fine_summary["confidence"],
        "margin": fine_summary["margin"],
        "relative_margin": fine_summary["relative_margin"],
        "fine_category_confident": fine_category_confident,
        "category_out_of_scope": category_out_of_scope,
        "score_distribution": fine_summary["score_distribution"],
        "parent_score_distribution": parent_summary["score_distribution"],
        "model": MODEL_NAME,
    }


def predict_category(image):
    """Compatibility wrapper returning only the fine category."""

    return predict_category_with_confidence(image)["category"]


def predict_semantic_colour(image):
    """Predict the garment's semantic colour."""

    ranked_prompts = _rank_prompt_predictions(
        image,
        COLOUR_PROMPTS,
    )

    ranked_colours = _aggregate_labels(
        ranked_prompts,
        COLOUR_PROMPTS,
    )

    summary = _prediction_summary(ranked_colours)

    return {
        "colour": summary["label"],
        "confidence": summary["confidence"],
        "margin": summary["margin"],
        "relative_margin": summary["relative_margin"],
        "score_distribution": summary["score_distribution"],
        "model": MODEL_NAME,
    }


def predict_style(image, maximum_styles=3, relative_threshold=0.35):
    """Predict several broad style cues rather than one exclusive class."""

    ranked_prompts = _rank_prompt_predictions(
        image,
        STYLE_PROMPTS,
    )

    ranked_styles = _aggregate_labels(
        ranked_prompts,
        STYLE_PROMPTS,
    )

    top_score = ranked_styles[0][1]
    selected_styles = []

    for style, score in ranked_styles:
        relative_score = score / top_score if top_score > 0 else 0.0

        if relative_score >= relative_threshold:
            selected_styles.append(
                {
                    "style": style,
                    "score": score,
                    "relative_score": relative_score,
                }
            )

        if len(selected_styles) >= maximum_styles:
            break

    if not selected_styles:
        style, score = ranked_styles[0]
        selected_styles.append(
            {
                "style": style,
                "score": score,
                "relative_score": 1.0,
            }
        )

    return {
        "primary_style": selected_styles[0]["style"],
        "styles": selected_styles,
        "confidence": selected_styles[0]["score"],
        "score_distribution": dict(ranked_styles),
        "model": MODEL_NAME,
    }
