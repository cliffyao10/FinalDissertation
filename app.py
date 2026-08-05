import io

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import predict_category
from src.colour_detection import predict_colour_details
from src.recommendation import recommend_outfit


FRAME_WIDTH = 520
FRAME_HEIGHT = 400
FRAME_BACKGROUND = (246, 247, 249)


st.set_page_config(
    page_title="AI Outfit Recommendation",
    page_icon="👗",
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
    """Run category, colour and recommendation modules."""

    category = predict_category(
        selected_image
    )

    colour_result = predict_colour_details(
        selected_image,
        reference_image=original_image,
    )

    colour = colour_result[
        "primary_colour"
    ]

    recommendation = recommend_outfit(
        category,
        colour,
    )

    return {
        "category": category,
        "colour": colour,
        "colour_result": colour_result,
        "recommendation": recommendation,
    }


def render_results(result):
    """Render compact review results."""

    category = result["category"]
    colour = result["colour"]
    colour_result = result["colour_result"]
    recommendation = result["recommendation"]

    with st.container(border=True):
        st.subheader("Recognition Result")

        category_column, colour_column = (
            st.columns(2)
        )

        with category_column:
            st.metric(
                "Category",
                category,
            )

        with colour_column:
            st.metric(
                "Colour",
                colour,
            )

        st.markdown(
            "#### Recommended Outfit"
        )

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

        st.markdown("#### Current Test Data")

        certainty_column, alternative_column = (
            st.columns(2)
        )

        with certainty_column:
            st.metric(
                "Colour certainty",
                (
                    f"{colour_result['confidence']:.0%}"
                ),
            )

        with alternative_column:
            st.metric(
                "Alternative",
                (
                    colour_result[
                        "secondary_colour"
                    ]
                    or "None"
                ),
            )

        with st.expander(
            "Detailed test data"
        ):
            st.write("Colour distribution")

            st.json(
                colour_result[
                    "colour_distribution"
                ]
            )

            st.write("White-balance gains")

            st.json(
                colour_result[
                    "white_balance_gains"
                ]
            )

            st.write("System methods")

            st.json(
                {
                    "category_recognition": (
                        "CLIP zero-shot"
                    ),
                    "colour_detection": (
                        colour_result["method"]
                    ),
                    "recommendation": (
                        recommendation["method"]
                    ),
                }
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
                "🛍️  PRODUCT IMAGE\n\n"
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
                "✏️  LIFESTYLE IMAGE\n\n"
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

            analyse_column, redraw_column = (
                st.columns(2)
            )

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
            st.session_state.analysis_result
        )