"""
Automatic colour detection module.

This module estimates the dominant colour of a clothing item.
It uses K-means clustering on the central area of the uploaded image
and maps the dominant RGB colour to a basic colour name.
"""

import colorsys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def rgb_to_colour_name(rgb):
    """
    Convert an RGB value into a basic colour name.

    Parameters:
        rgb (array-like): Red, green and blue values.

    Returns:
        str: Basic colour name.
    """

    red, green, blue = [int(value) for value in rgb]

    # Convert RGB from 0–255 to HSV values.
    hue, saturation, brightness = colorsys.rgb_to_hsv(
        red / 255,
        green / 255,
        blue / 255,
    )

    hue *= 360

    # Neutral colours
    if brightness < 0.18:
        return "Black"

    if saturation < 0.12 and brightness > 0.85:
        return "White"

    if saturation < 0.18:
        return "Grey"

    # Beige and brown require brightness checks.
    if 20 <= hue < 55:
        if brightness < 0.50:
            return "Brown"

        if saturation < 0.45 and brightness > 0.65:
            return "Beige"

    # Chromatic colours
    if hue < 15 or hue >= 345:
        return "Red"

    if hue < 40:
        return "Orange"

    if hue < 65:
        return "Yellow"

    if hue < 170:
        return "Green"

    if hue < 255:
        return "Blue"

    if hue < 290:
        return "Purple"

    if hue < 345:
        return "Pink"

    return "Unknown"


def predict_colour(image):
    """
    Estimate the dominant colour of an uploaded clothing image.

    The central part of the image is used to reduce the influence
    of the background.

    Parameters:
        image (PIL.Image.Image): Uploaded clothing image.

    Returns:
        str: Predicted basic colour name.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("The image must be a PIL Image.")

    # Ensure a consistent three-channel format.
    image = image.convert("RGB")

    width, height = image.size

    # Keep the central 60% of the image to reduce background influence.
    left = int(width * 0.20)
    top = int(height * 0.20)
    right = int(width * 0.80)
    bottom = int(height * 0.80)

    centre_image = image.crop((left, top, right, bottom))
    centre_image = centre_image.resize((100, 100))

    pixels = np.asarray(centre_image).reshape(-1, 3)

    # Find the three most important colour groups.
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(pixels)

    cluster_counts = np.bincount(labels)
    dominant_cluster = np.argmax(cluster_counts)
    dominant_rgb = model.cluster_centers_[dominant_cluster]

    colour_name = rgb_to_colour_name(dominant_rgb)

    return rgb_to_colour_name(dominant_rgb)