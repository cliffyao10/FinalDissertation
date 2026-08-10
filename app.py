import io

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import (
    extract_image_embedding,
    predict_category_with_confidence,
    predict_semantic_colour,
    predict_style,
)
from src.colour_detection import predict_colour_details
from src.recommendation import recommend_outfit
from src.weather import WeatherServiceError, get_city_weather


FRAME_WIDTH = 520
FRAME_HEIGHT = 400
FRAME_BACKGROUND = (246, 247, 249)

# Product mode shows recommendations only. Change this to True while
# collecting dissertation evidence or diagnosing recognition failures.
DEVELOPER_MODE = False

STYLE_OPTIONS = [
    "Casual",
    "Outdoor",
    "Sporty",
    "Formal",
    "Business",
    "Streetwear",
    "Minimalist",
    "Party",
    "Beachwear",
]

ICON_COLOURS = {
    "Black": "#26272b",
    "White": "#f7f7f5",
    "Grey": "#8b9098",
    "Blue": "#527db5",
    "Navy": "#293f62",
    "Red": "#bd4b51",
    "Green": "#66846a",
    "Olive": "#77744e",
    "Brown": "#87654e",
    "Beige": "#d7c4a5",
    "Cream": "#eee2c5",
    "Purple": "#806591",
    "Pink": "#d99aaa",
    "Orange": "#d4864a",
    "Yellow": "#d8ba4c",
}

CATEGORY_CHOICES = [
    "Tank Top",
    "T-Shirt",
    "Shirt",
    "Blouse",
    "Sweater",
    "Hoodie",
    "Jacket",
    "Blazer",
    "Coat",
    "Jeans",
    "Trousers",
    "Shorts",
    "Skirt",
    "Dress",
    "Bikini",
    "Swimsuit",
    "Shoes",
]

CATEGORY_RECOMMENDATION_ALIASES = {
    "Tank Top": "T-Shirt",
    "Blouse": "Shirt",
    "Blazer": "Jacket",
}

CATEGORY_PARENTS = {
    "Tank Top": "Top",
    "T-Shirt": "Top",
    "Shirt": "Top",
    "Blouse": "Top",
    "Sweater": "Top",
    "Hoodie": "Top",
    "Jacket": "Outerwear",
    "Blazer": "Outerwear",
    "Coat": "Outerwear",
    "Jeans": "Bottom",
    "Trousers": "Bottom",
    "Shorts": "Bottom",
    "Skirt": "Bottom",
    "Dress": "One-piece",
    "Bikini": "Swimwear",
    "Swimsuit": "Swimwear",
    "Shoes": "Footwear",
}


st.set_page_config(
    page_title="AI Outfit Recommendation",
    page_icon="👕",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }

        h1 {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }

        .result-item {
            min-height: 66px;
            padding: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid #d9eee0;
            border-radius: 12px;
            background: #f3faf5;
            color: #166534;
            font-weight: 600;
        }

        .empty-result {
            height: 400px;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px dashed #d1d5db;
            border-radius: 16px;
            background: #fafafa;
            color: #9ca3af;
        }

        .fixed-image-frame {
            width: 100%;
            max-width: 520px;
            height: 400px;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            background: #f6f7f9;
            overflow: hidden;
        }

        .stButton > button {
            min-height: 42px;
            border-radius: 11px;
            font-weight: 600;
        }

        div[data-testid="stFileUploader"] {
            margin-top: 0.35rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 110px;
            border-radius: 15px;
        }

        div[data-testid="stImage"] {
            max-width: 520px;
        }

        div[data-testid="stImage"] img {
            border-radius: 15px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 15px;
        }

        div[data-testid="stMetric"] {
            padding: 0;
        }
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(94vw, 1450px) !important;
            max-width: 1450px !important;
            height: min(92vh, 920px) !important;
            max-height: 92vh !important;
        }

        div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        div[data-testid="stDialog"] h2,
        div[data-testid="stDialog"] h3,
        div[data-testid="stDialog"] h4 {
            margin-top: 0.25rem !important;
            margin-bottom: 0.35rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialise_state():
    """Initialise application state."""

    defaults = {
        "image_mode": None,
        "analysis_result": None,
        "selected_image": None,
        "selection_ready": False,
        "canvas_version": 0,
        "uploader_version": 0,
        "uploaded_bytes": None,
        "uploaded_name": None,
        "city": "London",
        "city_input": "London",
        "weather_data": None,
        "weather_error": None,
    }

    for state_name, default_value in defaults.items():
        if state_name not in st.session_state:
            st.session_state[state_name] = default_value


def reset_analysis():
    """Clear selection and analysis data."""

    st.session_state.analysis_result = None
    st.session_state.selected_image = None
    st.session_state.selection_ready = False


def clear_uploaded_image():
    """Remove the uploaded image and reset the workflow."""

    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None
    st.session_state.uploader_version += 1
    st.session_state.canvas_version += 1

    reset_analysis()


def select_mode(mode):
    """Enter product or lifestyle mode."""

    st.session_state.image_mode = mode

    clear_uploaded_image()


def return_to_mode_selection():
    """Return to the opening page."""

    st.session_state.image_mode = None

    clear_uploaded_image()


def create_fixed_frame(
    image,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
):
    """
    Fit an image into a fixed-size frame without distortion.

    Returns:
        tuple: framed image and coordinate conversion information.
    """

    image = image.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        frame_width / original_width,
        frame_height / original_height,
    )

    displayed_width = max(
        1,
        int(original_width * scale),
    )

    displayed_height = max(
        1,
        int(original_height * scale),
    )

    resized_image = image.resize(
        (
            displayed_width,
            displayed_height,
        )
    )

    offset_x = (
        frame_width - displayed_width
    ) // 2

    offset_y = (
        frame_height - displayed_height
    ) // 2

    framed_image = Image.new(
        "RGB",
        (
            frame_width,
            frame_height,
        ),
        FRAME_BACKGROUND,
    )

    framed_image.paste(
        resized_image,
        (
            offset_x,
            offset_y,
        ),
    )

    frame_information = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "displayed_width": displayed_width,
        "displayed_height": displayed_height,
    }

    return framed_image, frame_information


def crop_from_drawing(
    original_image,
    drawing_object,
    frame_information,
):
    """
    Convert a freehand drawing boundary back to original-image
    coordinates and return the selected crop.
    """

    left = float(
        drawing_object.get("left", 0)
    )

    top = float(
        drawing_object.get("top", 0)
    )

    width = float(
        drawing_object.get("width", 0)
    )

    height = float(
        drawing_object.get("height", 0)
    )

    drawing_scale_x = float(
        drawing_object.get("scaleX", 1)
    )

    drawing_scale_y = float(
        drawing_object.get("scaleY", 1)
    )

    width *= drawing_scale_x
    height *= drawing_scale_y

    padding = 8

    drawing_left = left - padding
    drawing_top = top - padding
    drawing_right = left + width + padding
    drawing_bottom = top + height + padding

    image_left = frame_information["offset_x"]
    image_top = frame_information["offset_y"]

    image_right = (
        image_left
        + frame_information["displayed_width"]
    )

    image_bottom = (
        image_top
        + frame_information["displayed_height"]
    )

    # Keep the selection inside the displayed image rather than
    # including the frame's empty margins.
    drawing_left = max(
        drawing_left,
        image_left,
    )

    drawing_top = max(
        drawing_top,
        image_top,
    )

    drawing_right = min(
        drawing_right,
        image_right,
    )

    drawing_bottom = min(
        drawing_bottom,
        image_bottom,
    )

    if (
        drawing_right <= drawing_left
        or drawing_bottom <= drawing_top
    ):
        return None

    display_scale = frame_information["scale"]

    original_left = int(
        (drawing_left - image_left)
        / display_scale
    )

    original_top = int(
        (drawing_top - image_top)
        / display_scale
    )

    original_right = int(
        (drawing_right - image_left)
        / display_scale
    )

    original_bottom = int(
        (drawing_bottom - image_top)
        / display_scale
    )

    original_width, original_height = (
        original_image.size
    )

    original_left = max(
        0,
        min(original_left, original_width),
    )

    original_top = max(
        0,
        min(original_top, original_height),
    )

    original_right = max(
        0,
        min(original_right, original_width),
    )

    original_bottom = max(
        0,
        min(original_bottom, original_height),
    )

    if (
        original_right <= original_left
        or original_bottom <= original_top
    ):
        return None

    return original_image.crop(
        (
            original_left,
            original_top,
            original_right,
            original_bottom,
        )
    )


def analyse_clothing(
    selected_image,
    original_image,
    weather_data=None,
):
    """
    Use SigLIP as the primary recognition model.

    The pixel method is retained only as an interpretable
    baseline for debugging and dissertation comparison.
    """

    category_result = predict_category_with_confidence(
        selected_image
    )

    category = category_result["category"]
    recommendation_category = category_result.get(
        "recommendation_category",
        category,
    )

    style_result = predict_style(
        selected_image
    )

    pixel_colour_result = predict_colour_details(
        selected_image,
        reference_image=original_image,
    )

    semantic_colour_result = predict_semantic_colour(
        selected_image
    )

    input_embedding = extract_image_embedding(selected_image)

    pixel_colour = pixel_colour_result["primary_colour"]
    semantic_colour = semantic_colour_result["colour"]

    pixel_distribution = pixel_colour_result.get(
        "colour_distribution",
        {},
    )

    ranked_pixel_colours = sorted(
        pixel_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    colour_palette = [semantic_colour]

    # Black/Grey, White/Grey and White/Beige commonly describe shadows or
    # illumination on one neutral garment rather than intentional multicolour
    # design. Keep the palette conservative for these combinations.
    neutral_shading_pairs = [
        {"Black", "Grey"},
        {"White", "Grey"},
        {"White", "Beige"},
    ]

    # A second colour is retained only when the pixel method finds two
    # substantial regions and SigLIP agrees with one of those colours.
    if len(ranked_pixel_colours) >= 2:
        first_colour, first_share = ranked_pixel_colours[0]
        second_colour, second_share = ranked_pixel_colours[1]
        dominant_pair = {first_colour, second_colour}

        if (
            first_share >= 0.25
            and second_share >= 0.25
            and semantic_colour in dominant_pair
            and dominant_pair not in neutral_shading_pairs
        ):
            other_colour = (
                second_colour
                if semantic_colour == first_colour
                else first_colour
            )

            if other_colour != semantic_colour:
                colour_palette.append(other_colour)

    secondary_colour = (
        colour_palette[1]
        if len(colour_palette) > 1
        else None
    )

    models_agree = (
        pixel_colour == semantic_colour
    )

    siglip_margin = semantic_colour_result.get(
        "margin",
        0.0,
    )

    siglip_scores = sorted(
        (
            float(score)
            for score in semantic_colour_result.get(
                "score_distribution",
                {},
            ).values()
        ),
        reverse=True,
    )

    second_siglip_score = (
        siglip_scores[1]
        if len(siglip_scores) > 1
        else 0.0
    )

    relative_siglip_margin = (
        siglip_margin / siglip_scores[0]
        if siglip_scores and siglip_scores[0] > 0
        else 0.0
    )

    # The final product never interrupts the user with recognition choices.
    # SigLIP supplies the semantic colour; the pixel result remains available
    # only as an interpretable baseline in developer mode.
    final_colour = semantic_colour

    recognised_styles = [
        item["style"]
        for item in style_result.get("styles", [])
    ]

    recommendation = recommend_outfit(
        recommendation_category,
        final_colour,
        styles=recognised_styles,
        weather=weather_data,
        input_embedding=input_embedding,
    )

    if models_agree:
        confirmation_source = "model_agreement"
    elif relative_siglip_margin >= 0.35:
        confirmation_source = "siglip_clear_lead"
    else:
        confirmation_source = "siglip_automatic_fallback"

    return {
        "category": category,
        "recommendation_category": recommendation_category,
        "category_result": category_result,
        "style_result": style_result,
        "pixel_colour": pixel_colour,
        "pixel_colour_result": pixel_colour_result,
        "semantic_colour": semantic_colour,
        "semantic_colour_result": semantic_colour_result,
        "colour_palette": colour_palette,
        "secondary_colour": secondary_colour,
        "is_multicolour": len(colour_palette) > 1,
        "models_agree": models_agree,
        "colour": final_colour,
        "recommendation": recommendation,
        "confirmation_source": confirmation_source,
        "siglip_margin": siglip_margin,
        "siglip_relative_margin": relative_siglip_margin,
        "siglip_second_score": second_siglip_score,
        "weather": weather_data,
        "input_embedding": input_embedding,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_weather(city):
    """Cache a city's forecast for fifteen minutes."""

    return get_city_weather(city)


def analyse_with_current_weather(selected_image, original_image):
    """Run recognition with weather, falling back safely if unavailable."""

    try:
        weather_data = get_cached_weather(
            st.session_state.city.strip()
        )
        st.session_state.weather_data = weather_data
        st.session_state.weather_error = None
    except WeatherServiceError as error:
        weather_data = None
        st.session_state.weather_data = None
        st.session_state.weather_error = str(error)

    return analyse_clothing(
        selected_image,
        original_image,
        weather_data=weather_data,
    )


def recommendation_inputs(result):
    """Preserve model inputs when a user changes style, category or colour."""

    return {
        "styles": [
            item["style"]
            for item in result.get("style_result", {}).get("styles", [])
        ],
        "weather": result.get("weather"),
        "input_embedding": result.get("input_embedding"),
    }

def prepare_score_rows(score_distribution, limit=6):
    """Convert a score dictionary into compact sorted table rows."""

    if not score_distribution:
        return []

    sorted_scores = sorted(
        score_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    highest_score = float(sorted_scores[0][1]) if sorted_scores else 0.0

    return [
        {
            "Label": label,
            "Raw score": f"{float(score):.3%}",
            "Relative": (
                f"{float(score) / highest_score:.0%}"
                if highest_score > 0
                else "0%"
            ),
        }
        for label, score in sorted_scores[:limit]
    ]


@st.dialog(
    "Full Debug Report",
    width="large",
)
def show_debug_report(
    result,
    original_image,
    selected_image,
):
    """
    Display recognition inputs and intermediate results
    in one compact debugging window.
    """

    pixel_result = result["pixel_colour_result"]
    semantic_result = result["semantic_colour_result"]
    category_result = result["category_result"]
    style_result = result["style_result"]

    selected_image = (
        selected_image
        if selected_image is not None
        else original_image
    )

    original_preview, _ = create_fixed_frame(
        original_image,
        frame_width=340,
        frame_height=230,
    )

    selected_preview, _ = create_fixed_frame(
        selected_image,
        frame_width=340,
        frame_height=230,
    )

    image_column, summary_column = st.columns(
        [1.05, 1.25],
        gap="medium",
    )

    with image_column:
        original_column, selected_column = st.columns(2)

        with original_column:
            st.markdown("#### Original image")
            st.image(
                original_preview,
                use_container_width=True,
            )
            st.caption(
                f"Original size: "
                f"{original_image.width} × "
                f"{original_image.height}"
            )

        with selected_column:
            st.markdown("#### Analysed region")
            st.image(
                selected_preview,
                use_container_width=True,
            )
            st.caption(
                f"Crop size: "
                f"{selected_image.width} × "
                f"{selected_image.height}"
            )

        original_area = (
            original_image.width
            * original_image.height
        )

        selected_area = (
            selected_image.width
            * selected_image.height
        )

        crop_ratio = (
            selected_area / original_area
            if original_area
            else 0
        )

        st.caption(
            f"Selected area: {crop_ratio:.1%} "
            "of the original image"
        )

    with summary_column:
        st.markdown("#### Recognition summary")

        summary_one, summary_two, summary_three = (
            st.columns(3)
        )

        with summary_one:
            st.metric(
                "Category",
                result["category"],
            )

            if category_result.get("fine_category_confident", False):
                st.caption(
                    "Parent: "
                    f"{category_result.get('parent_category', 'Unknown')}"
                )
            elif result["category"] != "Unknown":
                likely_categories = category_result.get(
                    "likely_fine_categories",
                    [],
                )
                st.caption(
                    "Possible fine categories: "
                    + " / ".join(likely_categories[:2])
                )
            else:
                st.caption(
                    "Suggested category: "
                    f"{category_result.get('predicted_fine_category', 'Unknown')}"
                )

        with summary_two:
            st.metric(
                "Pixel colour",
                result["pixel_colour"],
            )

        with summary_three:
            st.metric(
                "SigLIP colour",
                result["semantic_colour"],
            )

        recognised_styles = ", ".join(
            style["style"]
            for style in style_result.get("styles", [])
        )

        st.caption(
            "Recognised style cues: "
            f"{recognised_styles or 'Unknown'}"
        )

        st.caption(
            "Colour palette: "
            + " + ".join(result.get("colour_palette", []))
        )

        summary_four, summary_five, summary_six = (
            st.columns(3)
        )

        with summary_four:
            st.metric(
                "Final colour",
                result["colour"] or "Unconfirmed",
            )

        with summary_five:
            st.metric(
                "Models agree",
                "Yes"
                if result["models_agree"]
                else "No",
            )

        with summary_six:
            st.metric(
                "Final source",
                result["confirmation_source"],
            )

        score_one, score_two, score_three = st.columns(3)

        with score_one:
            st.metric(
                "Category SigLIP score",
                (
                    f"{category_result.get('confidence', 0):.0%}"
                ),
            )

        with score_two:
            st.metric(
                "Pixel certainty",
                f"{pixel_result.get('confidence', 0):.0%}",
            )

        with score_three:
            st.metric(
                "SigLIP colour score",
                (
                    f"{semantic_result.get('confidence', 0):.0%}"
                ),
            )

        recommendation = result.get("recommendation")

        st.markdown("#### Recommendation state")

        if recommendation is None:
            if result["category"] == "Unknown":
                st.warning(
                    "Recommendation is waiting for category "
                    "confirmation."
                )
            else:
                st.warning(
                    "Recommendation is waiting for colour "
                    "confirmation."
                )
        else:
            recommended_items = " | ".join(
                recommendation.get("items", [])
            )

            st.success(recommended_items)

            st.caption(
                "Method: "
                f"{recommendation.get('method', 'Unknown')}"
            )

    st.divider()

    (
        category_column,
        style_column,
        colour_column,
        processing_column,
    ) = (
        st.columns(
            [1, 1, 1, 1],
            gap="medium",
        )
    )

    with category_column:
        st.markdown("#### Category SigLIP scores")

        category_rows = prepare_score_rows(
            category_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            category_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with style_column:
        st.markdown("#### SigLIP style scores")

        style_rows = prepare_score_rows(
            style_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            style_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with colour_column:
        st.markdown("#### SigLIP colour scores")

        semantic_rows = prepare_score_rows(
            semantic_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            semantic_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with processing_column:
        st.markdown("#### Pixel colour distribution")

        pixel_rows = prepare_score_rows(
            pixel_result.get(
                "colour_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            pixel_rows,
            use_container_width=True,
            hide_index=True,
            height=160,
        )

        gains = pixel_result.get(
            "white_balance_gains",
            {},
        )

        gain_rows = [
            {
                "Channel": channel.title(),
                "Gain": round(float(value), 3),
            }
            for channel, value in gains.items()
        ]

        st.markdown("#### White-balance gains")

        st.dataframe(
            gain_rows,
            use_container_width=True,
            hide_index=True,
            height=120,
        )

def render_results(
    result,
    original_image,
    selected_image,
):
    """
    Display category, two colour methods and recommendation results.
    """

    all_colours = [
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
    ]

    category = result["category"]
    category_result = result["category_result"]
    recommendation_category = result.get(
        "recommendation_category",
        category,
    )
    style_result = result["style_result"]
    pixel_colour = result["pixel_colour"]
    semantic_colour = result["semantic_colour"]

    pixel_result = result["pixel_colour_result"]
    semantic_result = result["semantic_colour_result"]

    with st.container(border=True):
        st.subheader("Recognition Result")

        category_column, style_column, colour_column = st.columns(3)

        with category_column:
            st.metric(
                "Category",
                category,
            )

            if category_result.get(
                "fine_category_confident",
                False,
            ):
                st.caption(
                    "Parent: "
                    f"{category_result.get('parent_category', 'Unknown')}"
                )
            elif category != "Unknown":
                likely_categories = category_result.get(
                    "likely_fine_categories",
                    [],
                )

                st.caption(
                    "Possible: "
                    + " / ".join(likely_categories[:2])
                )

                st.caption(
                    "Fine category uncertain; the broad category "
                    "is used for recommendation."
                )

        with style_column:
            st.metric(
                "Primary style",
                style_result["primary_style"],
            )

            secondary_styles = [
                item["style"]
                for item in style_result.get("styles", [])[1:]
            ]

            if secondary_styles:
                st.caption(
                    "Also: "
                    + ", ".join(secondary_styles)
                )

        with colour_column:
            final_colour = (
                result["colour"]
                if result["colour"] is not None
                else "Confirm below"
            )

            st.metric(
                "Final colour",
                final_colour,
            )

            colour_palette = result.get("colour_palette", [])

            if len(colour_palette) > 1:
                st.caption(
                    "Palette: "
                    + " + ".join(colour_palette)
                )

        st.markdown("#### Colour Comparison")

        pixel_column, semantic_column = st.columns(2)

        with pixel_column:
            st.metric(
                "Pixel method",
                pixel_colour,
                help=(
                    "White balance, Lab colour space "
                    "and K-means clustering."
                ),
            )

            st.caption(
                "Certainty score: "
                f"{pixel_result['confidence']:.0%}"
            )

        with semantic_column:
            st.metric(
                "SigLIP semantic method",
                semantic_colour,
                help=(
                    "Zero-shot visual-language "
                    "classification."
                ),
            )

            st.caption(
                "SigLIP score: "
                f"{semantic_result['confidence']:.0%}"
            )

        if st.button(
            "Open Full Debug Report",
            use_container_width=True,
            key="open_debug_report_button",
        ):
            show_debug_report(
                result,
                original_image,
                selected_image,
            )

        needs_category_confirmation = (
            category == "Unknown"
        )

        if needs_category_confirmation:
            predicted_fine_category = result[
                "category_result"
            ].get(
                "predicted_fine_category",
                "Unknown",
            )

            st.warning(
                "The garment does not strongly match the current "
                "category set. Please confirm its category."
            )

            suggested_index = (
                CATEGORY_CHOICES.index(predicted_fine_category)
                if predicted_fine_category in CATEGORY_CHOICES
                else 0
            )

            confirmed_category = st.selectbox(
                "Confirm clothing category",
                CATEGORY_CHOICES,
                index=suggested_index,
                key="unknown_category_choice",
            )

            if st.button(
                "Use Confirmed Category",
                type="primary",
                use_container_width=True,
                key="confirm_category_button",
            ):
                result["category"] = confirmed_category
                result["recommendation_category"] = (
                    CATEGORY_RECOMMENDATION_ALIASES.get(
                        confirmed_category,
                        confirmed_category,
                    )
                )
                result["category_result"][
                    "category_out_of_scope"
                ] = False
                result["category_result"][
                    "parent_category"
                ] = CATEGORY_PARENTS[confirmed_category]
                result["category_result"][
                    "fine_category_confident"
                ] = True
                result["category_confirmation_source"] = (
                    "user_confirmation"
                )

                if result["colour"] is not None:
                    result["recommendation"] = recommend_outfit(
                        result["recommendation_category"],
                        result["colour"],
                        **recommendation_inputs(result),
                    )

                st.session_state.analysis_result = result
                st.rerun()

            return

        needs_confirmation = (
            result["colour"] is None
        )

        if needs_confirmation:
            st.warning(
                "The colour result is uncertain. "
                "Please confirm the clothing colour."
            )

            confirmation_options = []

            for colour in [
                semantic_colour,
                pixel_colour,
                *all_colours,
            ]:
                if colour not in confirmation_options:
                    confirmation_options.append(colour)

            selected_colour = st.selectbox(
                "Confirm clothing colour",
                confirmation_options,
                key="colour_disagreement_choice",
            )

            if st.button(
                "Use Confirmed Colour",
                type="primary",
                use_container_width=True,
                key="confirm_colour_button",
            ):
                result["colour"] = selected_colour
                result["colour_palette"] = [selected_colour]
                result["secondary_colour"] = None
                result["is_multicolour"] = False

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    selected_colour,
                    **recommendation_inputs(result),
                )

                result["confirmation_source"] = (
                    "user_confirmation"
                )

                st.session_state.analysis_result = result

                st.rerun()

            # Do not show a recommendation until colour is confirmed.
            return

        recommendation = result["recommendation"]

        st.markdown("#### Recommended Outfit")

        st.caption(
            f"Method: {recommendation['method']}"
        )

        recommendation_columns = st.columns(
            len(recommendation["items"])
        )

        for column, item in zip(
            recommendation_columns,
            recommendation["items"],
        ):
            with column:
                st.markdown(
                    (
                        '<div class="result-item">'
                        f"{item}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        with st.expander("Why this works"):
            st.write(
                recommendation["explanation"]
            )

        with st.expander("Correct category manually"):
            predicted_fine_category = category_result.get(
                "predicted_fine_category",
                category,
            )

            category_index = (
                CATEGORY_CHOICES.index(predicted_fine_category)
                if predicted_fine_category in CATEGORY_CHOICES
                else 0
            )

            corrected_category = st.selectbox(
                "Correct category",
                CATEGORY_CHOICES,
                index=category_index,
                key="manual_category_override",
            )

            if st.button(
                "Apply Category Correction",
                use_container_width=True,
                key="apply_category_correction",
            ):
                corrected_recommendation_category = (
                    CATEGORY_RECOMMENDATION_ALIASES.get(
                        corrected_category,
                        corrected_category,
                    )
                )

                result["category"] = corrected_category
                result["recommendation_category"] = (
                    corrected_recommendation_category
                )
                result["category_result"][
                    "predicted_fine_category"
                ] = corrected_category
                result["category_result"][
                    "parent_category"
                ] = CATEGORY_PARENTS[corrected_category]
                result["category_result"][
                    "fine_category_confident"
                ] = True
                result["category_confirmation_source"] = (
                    "manual_correction"
                )
                result["recommendation"] = recommend_outfit(
                    corrected_recommendation_category,
                    result["colour"],
                    **recommendation_inputs(result),
                )

                st.session_state.analysis_result = result
                st.rerun()

        # Manual correction remains available even when models agree.
        with st.expander("Correct colour manually"):
            current_colour = result["colour"]

            current_index = all_colours.index(
                current_colour
            )

            corrected_colour = st.selectbox(
                "Correct colour",
                all_colours,
                index=current_index,
                key="manual_colour_override",
            )

            if st.button(
                "Apply Correction",
                use_container_width=True,
                key="apply_colour_correction",
            ):
                result["colour"] = corrected_colour
                result["colour_palette"] = [corrected_colour]
                result["secondary_colour"] = None
                result["is_multicolour"] = False

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    corrected_colour,
                    **recommendation_inputs(result),
                )

                result["confirmation_source"] = (
                    "manual_correction"
                )

                st.session_state.analysis_result = result

                st.rerun()

        st.markdown("#### Current Test Data")

        agreement_column, source_column = st.columns(2)

        with agreement_column:
            st.metric(
                "Models agree",
                "Yes" if result["models_agree"] else "No",
            )

        with source_column:
            st.metric(
                "Final source",
                result["confirmation_source"],
            )

        with st.expander("Detailed test data"):
            st.write("Pixel colour distribution")

            st.json(
                pixel_result["colour_distribution"]
            )

            st.write("SigLIP colour distribution")

            st.json(
                semantic_result["score_distribution"]
            )

            st.write("White-balance gains")

            st.json(
                pixel_result["white_balance_gains"]
            )

            st.write("Category SigLIP distribution")

            st.json(
                result["category_result"][
                    "score_distribution"
                ]
            )

            st.write("Style SigLIP distribution")

            st.json(
                style_result["score_distribution"]
            )


def garment_icon_svg(slot, colour, item_type=""):
    """Return a compact garment icon coloured to match the recommendation."""

    fill = ICON_COLOURS.get(colour, "#8b9098")
    stroke = "#34363d"

    paths = {
        "inner_top": (
            '<path d="M27 18 38 11h20l11 7 13 18-12 8-8-10v48H34V34l-8 10-12-8z"/>'
        ),
        "tank_top": (
            '<path d="M37 11h8c0 8 6 11 11 0h7l7 17-9 5v49H35V33l-8-5z"/>'
            '<path d="M43 12c0 9 10 9 12 0" fill="none"/>'
        ),
        "shorts": (
            '<path d="M31 15h34l3 48-17-1-3-20-3 20-17 1z"/>'
            '<path d="M31 28h34" fill="none"/>'
        ),
        "skirt": (
            '<path d="M36 14h24l13 66H23z"/>'
            '<path d="M34 27h28" fill="none"/>'
        ),
        "dress": (
            '<path d="M39 10h18l5 18 18 53H16l18-53z"/>'
            '<path d="M36 30h24" fill="none"/>'
        ),
        "no_layer": (
            '<circle cx="48" cy="48" r="18"/>'
            '<path d="M48 10v12M48 74v12M10 48h12M74 48h12M21 21l9 9M66 66l9 9M75 21l-9 9M30 66l-9 9" fill="none"/>'
        ),
        "outer_top": (
            '<path d="M28 17 40 10h16l12 7 14 20-12 7-8-12v50H34V32l-8 12-12-7z"/>'
            '<path d="M48 11v71M39 34h18" fill="none"/>'
        ),
        "bottom": (
            '<path d="M34 12h28l5 70H52l-4-44-4 44H29z"/>'
            '<path d="M34 25h28" fill="none"/>'
        ),
        "shoes": (
            '<path d="M17 53c13 0 19-8 25-18l12 15c5 6 12 9 24 10v14H17z"/>'
            '<path d="M53 68h25" fill="none"/>'
        ),
        "trainers": (
            '<path d="M15 54c14 0 21-7 28-20l12 15c6 7 13 10 27 12v13H15z"/>'
            '<path d="m43 44 15 13M37 51l9 7M16 67h66" fill="none"/>'
        ),
        "leather_shoes": (
            '<path d="M17 52c15 1 24-5 31-16l9 14c6 7 12 8 23 10v14H17z"/>'
            '<path d="M45 48h15M17 68h63" fill="none"/>'
        ),
        "boots": (
            '<path d="M30 13h28v42c5 4 12 6 23 7v13H26V55h4z"/>'
            '<path d="M31 52h28M26 68h55" fill="none"/>'
        ),
        "sandals": (
            '<path d="M18 63c15 0 24-5 33-17 8 8 17 12 29 15v13H18z"/>'
            '<path d="M39 53 53 68M54 50l12 16M18 68h62" fill="none"/>'
        ),
    }

    item_name = item_type.casefold()
    if "too hot" in item_name or "no outer" in item_name:
        icon_name = "no_layer"
    elif any(name in item_name for name in ("tank", "vest", "cami")):
        icon_name = "tank_top"
    elif "short" in item_name:
        icon_name = "shorts"
    elif "skirt" in item_name:
        icon_name = "skirt"
    elif "dress" in item_name and slot != "shoes":
        icon_name = "dress"
    elif "sandal" in item_name:
        icon_name = "sandals"
    elif "boot" in item_name:
        icon_name = "boots"
    elif any(
        name in item_name
        for name in ("loafer", "leather", "dress shoe")
    ):
        icon_name = "leather_shoes"
    elif any(
        name in item_name
        for name in ("trainer", "sneaker", "running", "trail")
    ):
        icon_name = "trainers"
    else:
        icon_name = slot

    path = paths.get(icon_name, paths["inner_top"])
    return (
        '<svg viewBox="0 0 96 96" width="76" height="76" '
        'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<g fill="{fill}" stroke="{stroke}" stroke-width="3" '
        f'stroke-linejoin="round">{path}</g></svg>'
    )


def render_outfit_cards(outfit):
    """Render one three-item outfit with coloured category icons."""

    columns = st.columns(3, gap="medium")

    for column, item in zip(columns, outfit["items"]):
        with column:
            st.markdown(
                (
                    '<div style="min-height:190px;padding:18px 12px;'
                    'border:1px solid #e2e6e3;border-radius:18px;'
                    'background:#fbfcfb;text-align:center">'
                    f'{garment_icon_svg(item["slot"], item["colour"], item["type"])}'
                    f'<div style="font-size:.78rem;color:#858a91;'
                    f'margin-top:4px">{item["slot_label"]}</div>'
                    f'<div style="font-weight:650;margin-top:7px">'
                    f'{item["type"]}</div>'
                    f'<div style="color:#666c73;margin-top:3px">'
                    f'{item["colour"]}</div></div>'
                ),
                unsafe_allow_html=True,
            )


def render_recommendations_only(result):
    """Render weather, recognised styles and two complete recommendations."""

    recommendation = result["recommendation"]

    # Rebuild results created before the four-slot recommender was loaded.
    if "primary" not in recommendation:
        recognised_styles = [
            item["style"]
            for item in result["style_result"].get("styles", [])
        ]
        recommendation = recommend_outfit(
            result["recommendation_category"],
            result["colour"],
            styles=recognised_styles,
            weather=result.get("weather"),
            input_embedding=result.get("input_embedding"),
        )
        result["recommendation"] = recommendation

    weather = result.get("weather")
    if weather:
        with st.container(border=True):
            st.subheader(f'{weather["city"]} Weather')
            weather_columns = st.columns(4)
            weather_columns[0].metric(
                "Temperature", f'{weather["temperature"]:.0f}°C'
            )
            weather_columns[1].metric(
                "Feels like", f'{weather["feels_like"]:.0f}°C'
            )
            weather_columns[2].metric(
                "Condition", weather["condition"]
            )
            weather_columns[3].metric(
                "Rain chance", f'{weather["rain_probability"]:.0f}%'
            )
    else:
        with st.container(border=True):
            st.subheader("Weather")
            st.caption(
                "Live weather is temporarily unavailable. The outfit uses "
                "colour and style information only."
            )

    recognised_styles = [
        item["style"]
        for item in result["style_result"].get("styles", [])
    ] or ["Casual"]

    with st.container(border=True):
        st.subheader("Style")
        st.caption(
            "Recognised: " + ", ".join(recognised_styles)
        )

        style_column, button_column = st.columns([1.35, 0.65])
        with style_column:
            selected_style = st.selectbox(
                "Choose a style",
                STYLE_OPTIONS,
                index=STYLE_OPTIONS.index(recognised_styles[0])
                if recognised_styles[0] in STYLE_OPTIONS
                else 0,
                key="recommendation_style_choice",
            )
        with button_column:
            st.write("")
            if st.button(
                "Recommend Again",
                use_container_width=True,
                type="primary",
            ):
                result["recommendation"] = recommend_outfit(
                    result["recommendation_category"],
                    result["colour"],
                    styles=recognised_styles,
                    weather=weather,
                    selected_style=selected_style,
                    input_embedding=result.get("input_embedding"),
                )
                st.session_state.analysis_result = result
                st.rerun()

    recommendation = result["recommendation"]

    with st.container(border=True):
        st.subheader(
            f'Primary Outfit · {recommendation["primary"]["style"]}'
        )
        if "model_score" in recommendation["primary"]:
            st.caption(
                "Lightweight model match: "
                f'{recommendation["primary"]["model_score"]:.1f}/100'
            )
        render_outfit_cards(recommendation["primary"])

    with st.container(border=True):
        st.subheader(
            f'Alternative Outfit · {recommendation["alternative"]["style"]}'
        )
        if "model_score" in recommendation["alternative"]:
            st.caption(
                "Lightweight model match: "
                f'{recommendation["alternative"]["model_score"]:.1f}/100'
            )
        render_outfit_cards(recommendation["alternative"])


initialise_state()


st.title("AI Outfit Recommendation")


# =========================================================
# Opening mode-selection page
# =========================================================

if st.session_state.image_mode is None:
    st.markdown(
        """
        <style>
            .stButton > button {
                min-height: 165px;
                padding: 1.5rem 1.7rem;
                border: 1px solid #e5e7eb;
                border-radius: 22px;
                background: #ffffff;
                box-shadow:
                    0 8px 26px
                    rgba(15, 23, 42, 0.06);
                font-size: 1.05rem;
                line-height: 1.55;
                text-align: left;
                white-space: pre-wrap;
                transition:
                    transform 0.15s ease,
                    border-color 0.15s ease,
                    box-shadow 0.15s ease;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                border-color: #ef5b5b;
                color: #111827;
                box-shadow:
                    0 12px 32px
                    rgba(15, 23, 42, 0.10);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("weather_city_form", border=False):
        city_column, save_city_column = st.columns([0.78, 0.22])
        with city_column:
            st.text_input(
                "City for today's weather",
                key="city_input",
                placeholder="For example: Shanghai",
                help=(
                    "Press Enter or choose Save City. The city is used for "
                    "today's temperature, rain and wind."
                ),
            )
        with save_city_column:
            st.write("")
            city_submitted = st.form_submit_button(
                "Save City",
                use_container_width=True,
            )

        if city_submitted:
            entered_city = st.session_state.city_input.strip()
            if entered_city:
                st.session_state.city = entered_city
                st.session_state.weather_data = None
                st.session_state.weather_error = None

    st.caption(f'Weather location: **{st.session_state.city}**')

    product_column, lifestyle_column = (
        st.columns(
            2,
            gap="large",
        )
    )

    with product_column:
        if st.button(
            (
                "PRODUCT IMAGE\n\n"
                "Use a clean catalogue or product "
                "photo. The complete image will "
                "be analysed."
            ),
            use_container_width=True,
            key="product_mode_button",
        ):
            select_mode("product")
            st.rerun()

    with lifestyle_column:
        if st.button(
            (
                "LIFESTYLE IMAGE\n\n"
                "Upload a person or lifestyle "
                "photo and quickly draw around "
                "one clothing item."
            ),
            use_container_width=True,
            key="lifestyle_mode_button",
        ):
            select_mode("lifestyle")
            st.rerun()

    st.stop()


# =========================================================
# Working page
# =========================================================

mode_title = (
    "Product Image"
    if st.session_state.image_mode
    == "product"
    else "Lifestyle Image"
)


left_column, right_column = st.columns(
    [0.95, 1.05],
    gap="large",
)


# ---------------------------------------------------------
# Left-column navigation
# ---------------------------------------------------------

with left_column:
    navigation_column, title_column = (
        st.columns(
            [0.40, 0.60],
            vertical_alignment="center",
        )
    )

    with navigation_column:
        if st.button(
            "← Back",
            use_container_width=True,
            key="back_button",
        ):
            return_to_mode_selection()
            st.rerun()

    with title_column:
        st.subheader(mode_title)


# ---------------------------------------------------------
# Upload image only when no image is stored
# ---------------------------------------------------------

if st.session_state.uploaded_bytes is None:
    with left_column:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=(
                "uploader_"
                f"{st.session_state.image_mode}_"
                f"{st.session_state.uploader_version}"
            ),
        )

        if uploaded_file is not None:
            st.session_state.uploaded_bytes = (
                uploaded_file.getvalue()
            )

            st.session_state.uploaded_name = (
                uploaded_file.name
            )

            st.session_state.canvas_version += 1

            reset_analysis()

            st.rerun()

    with right_column:
        st.markdown(
            """
            <div class="empty-result">
                Upload an image to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ---------------------------------------------------------
# Load stored original image
# ---------------------------------------------------------

original_image = Image.open(
    io.BytesIO(
        st.session_state.uploaded_bytes
    )
).convert("RGB")


# ---------------------------------------------------------
# Left-column image controls
# ---------------------------------------------------------

with left_column:
    if st.button(
        "↻ Change Image",
        use_container_width=True,
        key="change_image_button",
    ):
        clear_uploaded_image()
        st.rerun()

    if st.session_state.image_mode == "product":
        framed_product, _ = create_fixed_frame(
            original_image
        )

        st.image(
            framed_product,
            width=FRAME_WIDTH,
        )

        st.session_state.selected_image = (
            original_image
        )

        st.session_state.selection_ready = True

        if st.button(
            "Analyse Item",
            type="primary",
            use_container_width=True,
            key="analyse_product_button",
        ):
            with st.spinner("Analysing..."):
                st.session_state.analysis_result = (
                    analyse_with_current_weather(
                        original_image,
                        original_image,
                    )
                )

            st.rerun()

    else:
        # -------------------------------------------------
        # Lifestyle mode: original canvas
        # -------------------------------------------------

        if not st.session_state.selection_ready:
            canvas_frame, frame_information = (
                create_fixed_frame(
                    original_image
                )
            )

            canvas_result = st_canvas(
                background_image=canvas_frame,
                drawing_mode="freedraw",
                stroke_width=6,
                stroke_color="#FF4B4B",
                fill_color=(
                    "rgba(255, 75, 75, 0.10)"
                ),
                update_streamlit=True,
                display_toolbar=True,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                key=(
                    "canvas_"
                    f"{st.session_state.uploaded_name}_"
                    f"{st.session_state.canvas_version}"
                ),
            )

            canvas_objects = []

            if canvas_result.json_data is not None:
                canvas_objects = (
                    canvas_result.json_data.get(
                        "objects",
                        [],
                    )
                )

            if canvas_objects:
                latest_drawing = (
                    canvas_objects[-1]
                )

                selected_image = (
                    crop_from_drawing(
                        original_image,
                        latest_drawing,
                        frame_information,
                    )
                )

                if selected_image is not None:
                    st.session_state.selected_image = (
                        selected_image
                    )

                    st.session_state.selection_ready = (
                        True
                    )

                    st.session_state.analysis_result = (
                        None
                    )

                    st.rerun()

                else:
                    st.warning(
                        "Please draw around the "
                        "clothing item itself."
                    )

        # -------------------------------------------------
        # Lifestyle mode: selected item only
        # -------------------------------------------------

        else:
            selected_image = (
                st.session_state.selected_image
            )

            framed_selection, _ = (
                create_fixed_frame(
                    selected_image
                )
            )

            st.image(
                framed_selection,
                width=FRAME_WIDTH,
            )

            redraw_column, analyse_column = (
                st.columns(2)
            )

            with redraw_column:
                if st.button(
                    "Draw Again",
                    use_container_width=True,
                    key="draw_again_button",
                ):
                    st.session_state.selected_image = (
                        None
                    )

                    st.session_state.selection_ready = (
                        False
                    )

                    st.session_state.analysis_result = (
                        None
                    )

                    st.session_state.canvas_version += (
                        1
                    )

                    st.rerun()

            with analyse_column:
                if st.button(
                    "Analyse Item",
                    type="primary",
                    use_container_width=True,
                    key=(
                        "analyse_lifestyle_button"
                    ),
                ):
                    with st.spinner(
                        "Analysing..."
                    ):
                        st.session_state.analysis_result = (
                            analyse_with_current_weather(
                                selected_image,
                                original_image,
                            )
                        )

                    st.rerun()


# ---------------------------------------------------------
# Right-column results
# ---------------------------------------------------------

with right_column:
    if st.session_state.analysis_result is None:
        st.markdown(
            """
            <div class="empty-result">
                Select and analyse one clothing
                item to view the recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        if DEVELOPER_MODE:
            render_results(
                st.session_state.analysis_result,
                original_image,
                st.session_state.selected_image,
            )
        else:
            render_recommendations_only(
                st.session_state.analysis_result
            )
