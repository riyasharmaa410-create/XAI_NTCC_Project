
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Image Analysis | XAI",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Analysis")
st.write(
    "Upload an image to begin image analysis. "
    "Explainable AI features will help explain a model's prediction."
)

st.info(
    "This is the initial image-upload page. "
    "Image predictions and visual explanations will be added next."
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Choose a PNG or JPG image from your device."
)

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

        st.success("Image uploaded successfully!")

    except Exception:
        st.error("Unable to open this image. Please upload a valid PNG or JPG file.")

