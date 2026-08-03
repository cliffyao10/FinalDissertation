import streamlit as st
from PIL import Image

from src.recognition import predict_category
from src.colour_detection import predict_colour
from src.recommendation import recommend_outfit


st.set_page_config(
    page_title="AI Outfit Recommendation Prototype",
    layout="centered"
)

st.title("AI Outfit Recommendation Prototype")
st.caption("Development version: automatic colour detection")

st.write(
    "Upload a clothing image. The system will identify the item category "
    "and colour, then generate a simple outfit recommendation."
)

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded clothing image", use_container_width=True)

    category = predict_category(image)
    colour = predict_colour(image)

    st.subheader("Recognition Result")
    st.write(f"Category: **{category}**")
    st.write(f"Colour: **{colour}**")

    recommendation = recommend_outfit(category, colour)

    st.subheader("Recommended Outfit")

    for item in recommendation["items"]:
        st.success(item)

    st.subheader("Recommendation Explanation")
    st.write(recommendation["explanation"])

    st.info(
        "The current prototype uses CLIP zero-shot category recognition, "
    "automatic dominant colour detection, and a general rule-based "
    "outfit recommendation baseline."
    )

else:
    st.info("Please upload a clothing image to start.")
    