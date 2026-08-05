import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import predict_category
from src.colour_detection import predict_colour_details
from src.recommendation import recommend_outfit


st.set_page_config(
    page_title="AI Outfit Recommendation Prototype",
    layout="centered",
)

st.title("AI Outfit Recommendation Prototype")

st.write(
    "Upload a product image or quickly draw around one clothing item "
    "in a lifestyle image."
)


def resize_for_canvas(image, maximum_width=700):
    """
    Resize an image for display while preserving its aspect ratio.

    Returns:
        tuple: resized image, width and height.
    """

    width, height = image.size

    if width <= maximum_width:
        return image, width, height

    scale = maximum_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = image.resize((new_width, new_height))

    return resized_image, new_width, new_height


def crop_from_drawing(
    original_image,
    drawing_object,
    canvas_width,
    canvas_height,
):
    """
    Crop the original image using the bounding area of a freehand drawing.
    """

    left = float(drawing_object.get("left", 0))
    top = float(drawing_object.get("top", 0))
    width = float(drawing_object.get("width", 0))
    height = float(drawing_object.get("height", 0))

    scale_x = float(drawing_object.get("scaleX", 1))
    scale_y = float(drawing_object.get("scaleY", 1))

    width *= scale_x
    height *= scale_y

    # Add a small margin around the user's drawing.
    padding = 10

    canvas_left = max(0, left - padding)
    canvas_top = max(0, top - padding)
    canvas_right = min(canvas_width, left + width + padding)
    canvas_bottom = min(canvas_height, top + height + padding)

    original_width, original_height = original_image.size

    horizontal_scale = original_width / canvas_width
    vertical_scale = original_height / canvas_height

    original_left = int(canvas_left * horizontal_scale)
    original_top = int(canvas_top * vertical_scale)
    original_right = int(canvas_right * horizontal_scale)
    original_bottom = int(canvas_bottom * vertical_scale)

    return original_image.crop(
        (
            original_left,
            original_top,
            original_right,
            original_bottom,
        )
    )


input_mode = st.radio(
    "Choose image type",
    [
        "Product image",
        "Lifestyle or person image",
    ],
)

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"],
)

selected_image = None


if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    if input_mode == "Product image":
        st.subheader("Uploaded Product Image")

        st.image(
            original_image,
            caption="The complete product image will be analysed.",
            use_container_width=True,
        )

        selected_image = original_image

    else:
        st.subheader("Circle One Clothing Item")

        st.write(
            "Use the mouse to quickly draw around the clothing item. "
            "The system will analyse the area covered by your drawing."
        )

        canvas_image, canvas_width, canvas_height = resize_for_canvas(
            original_image
        )

        canvas_result = st_canvas(
            background_image=canvas_image,
            drawing_mode="freedraw",
            stroke_width=6,
            stroke_color="#FF0000",
            fill_color="rgba(255, 0, 0, 0.10)",
            update_streamlit=True,
            display_toolbar=True,
            width=canvas_width,
            height=canvas_height,
            key=f"clothing_canvas_{uploaded_file.name}",
        )

        if (
            canvas_result.json_data is not None
            and canvas_result.json_data.get("objects")
        ):
            drawing_objects = canvas_result.json_data["objects"]

            # Use the most recent drawing.
            latest_drawing = drawing_objects[-1]

            selected_image = crop_from_drawing(
                original_image,
                latest_drawing,
                canvas_width,
                canvas_height,
            )

            st.subheader("Selected Clothing Preview")

            st.image(
                selected_image,
                caption="Only this region will be analysed.",
                use_container_width=True,
            )

        else:
            st.info("Draw around one clothing item to continue.")

    if selected_image is not None:
        if st.button(
            "Analyse this clothing item",
            type="primary",
        ):
            with st.spinner("Analysing the selected clothing item..."):
                category = predict_category(selected_image)
                colour_result = predict_colour_details(
                    selected_image,
                    reference_image=original_image,
                )
                colour = colour_result["primary_colour"]
                recommendation = recommend_outfit(category, colour)

            st.subheader("Recognition Result")

            st.write(f"Category: **{category}**")
            st.write(f"Colour: **{colour}**")
            st.write(
                f"Colour confidence: **{colour_result['confidence']:.0%}**"
            )

            if colour_result["secondary_colour"] is not None:
                st.write(
                    "Alternative colour: "
                    f"**{colour_result['secondary_colour']}**"
                )

            with st.expander("Colour analysis details"):
                st.write(
                    "Colour distribution:",
                    colour_result["colour_distribution"],
                )
                st.write(
                    "White-balance gains:",
                    colour_result["white_balance_gains"],
                )

            st.subheader("Recommended Outfit")

            st.caption(
                f"Recommendation method: {recommendation['method']}"
            )

            for item in recommendation["items"]:
                st.success(item)

            st.subheader("Recommendation Explanation")

            st.write(recommendation["explanation"])

            st.info(
                "The prototype uses CLIP zero-shot category recognition, "
                "automatic colour detection, interactive region selection, "
                "and a general rule-based recommendation baseline."
            )

else:
    st.info("Please upload an image to start.")