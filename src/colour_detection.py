"""
General clothing colour detection module.

The module performs bounded colour correction, extracts dominant
colour clusters, and maps them to basic colour names in CIE Lab space.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# Standard colour prototypes represented in RGB.
COLOUR_PROTOTYPES_RGB = {
    "Black": (20, 20, 20),
    "White": (240, 240, 240),
    "Grey": (128, 128, 128),
    "Red": (200, 40, 40),
    "Orange": (230, 120, 30),
    "Yellow": (230, 205, 40),
    "Green": (50, 145, 70),
    "Blue": (50, 90, 190),
    "Purple": (125, 70, 160),
    "Pink": (225, 130, 165),
    "Brown": (115, 75, 45),
    "Beige": (210, 190, 150),
}


def rgb_to_lab(rgb_values):
    """
    Convert RGB values to CIE Lab.

    Parameters:
        rgb_values: Array shaped (..., 3), using values from 0 to 255.

    Returns:
        numpy.ndarray: Lab values with the same leading dimensions.
    """

    rgb = np.asarray(rgb_values, dtype=np.float32) / 255.0

    # Convert sRGB to linear RGB.
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # Linear RGB to CIE XYZ using the D65 reference illuminant.
    conversion_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )

    xyz = linear_rgb @ conversion_matrix.T

    # D65 reference white.
    reference_white = np.array(
        [0.95047, 1.00000, 1.08883],
        dtype=np.float32,
    )

    xyz = xyz / reference_white

    delta = 6 / 29
    threshold = delta ** 3

    transformed_xyz = np.where(
        xyz > threshold,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4 / 29,
    )

    x_value = transformed_xyz[..., 0]
    y_value = transformed_xyz[..., 1]
    z_value = transformed_xyz[..., 2]

    lightness = 116 * y_value - 16
    green_red = 500 * (x_value - y_value)
    blue_yellow = 200 * (y_value - z_value)

    return np.stack(
        [lightness, green_red, blue_yellow],
        axis=-1,
    )


PROTOTYPE_NAMES = list(COLOUR_PROTOTYPES_RGB.keys())

PROTOTYPE_LAB = rgb_to_lab(
    np.array(
        list(COLOUR_PROTOTYPES_RGB.values()),
        dtype=np.float32,
    )
)


def estimate_white_balance_gains(reference_image):
    """
    Estimate conservative channel gains from the complete source image.

    The correction is deliberately limited so a single-colour garment
    cannot be incorrectly neutralised.
    """

    reference = reference_image.convert("RGB")
    reference.thumbnail((300, 300))

    pixels = np.asarray(reference, dtype=np.float32).reshape(-1, 3)

    # Estimate the bright response of each colour channel.
    bright_values = np.percentile(pixels, 95, axis=0)

    bright_values = np.maximum(bright_values, 1.0)
    target_value = float(np.mean(bright_values))

    gains = target_value / bright_values

    # Prevent aggressive corrections in strongly coloured scenes.
    gains = np.clip(gains, 0.80, 1.25)

    return gains


def apply_white_balance(image, gains):
    """Apply channel gains to a PIL image."""

    pixels = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    corrected_pixels = pixels * gains.reshape(1, 1, 3)
    corrected_pixels = np.clip(corrected_pixels, 0, 255).astype(np.uint8)

    return Image.fromarray(corrected_pixels, mode="RGB")


def crop_central_region(image):
    """
    Keep the central 80% of the selected region.

    This reduces small amounts of background that may remain around
    the user's freehand selection.
    """

    width, height = image.size

    left = int(width * 0.10)
    top = int(height * 0.10)
    right = int(width * 0.90)
    bottom = int(height * 0.90)

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


def map_rgb_to_colour(rgb_value):
    """
    Map one RGB cluster centre to the nearest standard colour in Lab.

    Returns:
        tuple: colour name and Lab distance.
    """

    cluster_lab = rgb_to_lab(
        np.asarray(rgb_value, dtype=np.float32)
    )

    distances = np.linalg.norm(
        PROTOTYPE_LAB - cluster_lab,
        axis=1,
    )

    nearest_index = int(np.argmin(distances))

    return (
        PROTOTYPE_NAMES[nearest_index],
        float(distances[nearest_index]),
    )


def predict_colour_details(image, reference_image=None):
    """
    Detect the main colour and return detailed evidence.

    Parameters:
        image (PIL.Image.Image): Selected clothing region.
        reference_image (PIL.Image.Image | None):
            Complete source image used to estimate illumination.

    Returns:
        dict: Primary colour, alternative colour, confidence,
        colour distribution and diagnostic information.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("The image must be a PIL Image.")

    if reference_image is None:
        reference_image = image

    if not isinstance(reference_image, Image.Image):
        raise TypeError("The reference image must be a PIL Image.")

    gains = estimate_white_balance_gains(reference_image)

    corrected_image = apply_white_balance(
        image.convert("RGB"),
        gains,
    )

    analysis_image = crop_central_region(corrected_image)
    analysis_image = analysis_image.resize((120, 120))

    pixels = np.asarray(
        analysis_image,
        dtype=np.float32,
    ).reshape(-1, 3)

    number_of_clusters = min(4, len(np.unique(pixels, axis=0)))

    if number_of_clusters < 1:
        raise ValueError("The selected image does not contain usable pixels.")

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_labels = model.fit_predict(pixels)
    cluster_counts = np.bincount(cluster_labels)

    total_pixels = cluster_counts.sum()

    colour_weights = {}
    colour_distances = {}

    for cluster_index, cluster_centre in enumerate(
        model.cluster_centers_
    ):
        colour_name, colour_distance = map_rgb_to_colour(
            cluster_centre
        )

        cluster_weight = (
            float(cluster_counts[cluster_index]) / total_pixels
        )

        colour_weights[colour_name] = (
            colour_weights.get(colour_name, 0.0)
            + cluster_weight
        )

        if colour_name not in colour_distances:
            colour_distances[colour_name] = colour_distance
        else:
            colour_distances[colour_name] = min(
                colour_distances[colour_name],
                colour_distance,
            )

    ranked_colours = sorted(
        colour_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_colour, primary_weight = ranked_colours[0]

    if len(ranked_colours) > 1:
        secondary_colour, secondary_weight = ranked_colours[1]
    else:
        secondary_colour = None
        secondary_weight = 0.0

    primary_distance = colour_distances[primary_colour]

    # Confidence combines pixel dominance and prototype similarity.
    distance_confidence = max(
        0.0,
        1.0 - primary_distance / 60.0,
    )

    confidence = (
        0.65 * primary_weight
        + 0.35 * distance_confidence
    )

    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        "primary_colour": primary_colour,
        "secondary_colour": secondary_colour,
        "confidence": confidence,
        "colour_distribution": {
            colour_name: round(weight, 3)
            for colour_name, weight in ranked_colours
        },
        "white_balance_gains": {
            "red": round(float(gains[0]), 3),
            "green": round(float(gains[1]), 3),
            "blue": round(float(gains[2]), 3),
        },
        "method": "bounded_white_balance_lab_clustering",
    }


def predict_colour(image, reference_image=None):
    """
    Compatibility wrapper returning only the primary colour name.
    """

    result = predict_colour_details(
        image,
        reference_image=reference_image,
    )

    return result["primary_colour"]