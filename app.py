import streamlit as st
from PIL import Image

st.title("AI Outfit Recommendation Prototype")
st.write("Environment setup successful.")

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    st.subheader("Recognition Result")
    st.write("Category: Jeans")
    st.write("Colour: Black")

    st.subheader("Recommended Outfit")
    st.write("1. White T-shirt")
    st.write("2. Grey jacket")
    st.write("3. White trainers")
else:
    st.info("Please upload a clothing image.")