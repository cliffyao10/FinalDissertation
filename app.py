import io

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import (
    predict_category_with_confidence,
    predict_semantic_colour,
    predict_style,
)
from src.colour_detection import predict_colour_details
from src.recommendation import recommend_outfit


FRAME_WIDTH = 520
FRAME_HEIGHT = 400
FRAME_BACKGROUND = (246, 247, 249)


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

    pixel_colour = pixel_colour_result["primary_colour"]
    semantic_colour = semantic_colour_result["colour"]

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

    # Agreement is accepted immediately. When the methods
    # disagree, SigLIP is accepted only if its first result
    # has a clear relative lead over its second result.
    minimum_relative_margin = 0.35

    if models_agree:
        final_colour = semantic_colour

        recommendation = recommend_outfit(
            recommendation_category,
            final_colour,
        )

        confirmation_source = "model_agreement"

    elif relative_siglip_margin >= minimum_relative_margin:
        final_colour = semantic_colour

        recommendation = recommend_outfit(
            recommendation_category,
            final_colour,
        )

        confirmation_source = "siglip_clear_lead"

    else:
        final_colour = None
        recommendation = None
        confirmation_source = "awaiting_user"

    return {
        "category": category,
        "recommendation_category": recommendation_category,
        "category_result": category_result,
        "style_result": style_result,
        "pixel_colour": pixel_colour,
        "pixel_colour_result": pixel_colour_result,
        "semantic_colour": semantic_colour,
        "semantic_colour_result": semantic_colour_result,
        "models_agree": models_agree,
        "colour": final_colour,
        "recommendation": recommendation,
        "confirmation_source": confirmation_source,
        "siglip_margin": siglip_margin,
        "siglip_relative_margin": relative_siglip_margin,
        "siglip_second_score": second_siglip_score,
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

    return [
        {
            "Label": label,
            "Score": f"{float(score):.1%}",
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

            st.caption(
                "Parent: "
                f"{category_result.get('parent_category', 'Unknown')}"
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

            st.caption(
                "Parent: "
                f"{result['category_result'].get('parent_category', 'Unknown')}"
            )

            if not result["category_result"].get(
                "fine_category_confident",
                True,
            ):
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
                or "Confirm below"
            )

            st.metric(
                "Final colour",
                final_colour,
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

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    selected_colour,
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

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    corrected_colour,
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
                    analyse_clothing(
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
                            analyse_clothing(
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
        render_results(
            st.session_state.analysis_result,
            original_image,
            st.session_state.selected_image,
        )
