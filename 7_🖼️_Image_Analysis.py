
import streamlit as st
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.models import ResNet18_Weights


st.set_page_config(
    page_title="Image Analysis | XAI",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Analysis")
st.write(
    "Classify an image and understand which image regions "
    "influenced the model's prediction using Grad-CAM."
)


@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights


uploaded_image = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of a common object."
)


if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, caption="Original image", width="stretch")

        with st.spinner("Loading model and analyzing image..."):
            model, weights = load_model()
            input_tensor = weights.transforms()(image).unsqueeze(0)

            activations = {}
            gradients = {}

            def save_activation(module, inputs, output):
                activations["value"] = output

            def save_gradient(module, grad_input, grad_output):
                gradients["value"] = grad_output[0]

            target_layer = model.layer4[-1]

            forward_handle = target_layer.register_forward_hook(
                save_activation
            )
            backward_handle = target_layer.register_full_backward_hook(
                save_gradient
            )

            try:
                with torch.enable_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)

                    class_index = int(
                        output.argmax(dim=1).item()
                    )

                    model.zero_grad()
                    output[0, class_index].backward()

            finally:
                forward_handle.remove()
                backward_handle.remove()

            # Main prediction
            confidence = (
                probabilities[0, class_index].item() * 100
            )
            class_name = weights.meta["categories"][class_index]

            # Top 5 predictions
            top5_probabilities, top5_indices = torch.topk(
                probabilities[0], k=5
            )

            top5_predictions = [
                {
                    "Rank": rank,
                    "Class": weights.meta["categories"][int(index)],
                    "Confidence (%)": round(
                        float(probability) * 100, 2
                    )
                }
                for rank, (probability, index) in enumerate(
                    zip(
                        top5_probabilities.tolist(),
                        top5_indices.tolist()
                    ),
                    start=1
                )
            ]

            # Grad-CAM calculation
            feature_maps = activations["value"]
            grads = gradients["value"]

            channel_weights = grads.mean(
                dim=(2, 3), keepdim=True
            )

            cam = (
                channel_weights * feature_maps
            ).sum(dim=1, keepdim=True)

            cam = torch.relu(cam)

            cam = torch.nn.functional.interpolate(
                cam,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0].detach().cpu().numpy()

            cam_min = cam.min()
            cam_max = cam.max()

            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

        # Display prediction
        st.subheader("Prediction")
        st.success(f"Predicted class: **{class_name}**")

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "This pretrained model recognizes objects from ImageNet's "
            "1000 categories. Confidence is not a guarantee of correctness."
        )

        # Display Top 5
        st.subheader("Top 5 Predictions")

        st.dataframe(
            top5_predictions,
            hide_index=True,
            width="stretch"
        )

        # Display explanation
        st.subheader("Grad-CAM Explanation")

        st.write(
            "The heatmap highlights image regions that influenced "
            "the selected class score. It provides an approximate "
            "visual explanation, not proof of correct reasoning."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original image",
                width="stretch"
            )

        with col2:
            fig, ax = plt.subplots()

            ax.imshow(image)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

    except Exception as error:
        st.error("Image analysis failed.")

        with st.expander("View error details"):
            st.code(str(error))
