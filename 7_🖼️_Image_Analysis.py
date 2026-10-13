
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


@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights


def generate_gradcam(model, input_tensor, class_index):
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
        model.zero_grad()

        with torch.enable_grad():
            output = model(input_tensor)
            output[0, class_index].backward()

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
            size=(input_tensor.shape[2], input_tensor.shape[3]),
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

        return cam

    finally:
        forward_handle.remove()
        backward_handle.remove()


st.title("🖼️ Image Analysis")
st.write(
    "Classify an image, compare the top predictions, and visualize "
    "which image regions influenced the selected prediction."
)

st.info(
    "This page uses a pretrained ResNet18 model trained on ImageNet. "
    "It recognizes common objects across 1,000 categories."
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Choose a clear image of a common object."
)


if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Original uploaded image",
            width="stretch"
        )

        with st.spinner(
            "Loading the model and analyzing your image..."
        ):
            model, weights = load_model()

            input_tensor = weights.transforms()(image).unsqueeze(0)

            # Run image classification
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]

            # Find the five most likely classes
            top5_probabilities, top5_indices = torch.topk(
                probabilities, k=5
            )

            class_index = int(top5_indices[0].item())
            class_name = weights.meta["categories"][class_index]

            confidence = (
                float(top5_probabilities[0].item()) * 100
            )

            top5_predictions = []

            for rank, (probability, index) in enumerate(
                zip(
                    top5_probabilities.tolist(),
                    top5_indices.tolist()
                ),
                start=1
            ):
                top5_predictions.append({
                    "Rank": rank,
                    "Class": weights.meta["categories"][int(index)],
                    "Confidence (%)": round(
                        float(probability) * 100, 2
                    )
                })

            # Generate Grad-CAM for the highest-ranked class
            cam = generate_gradcam(
                model,
                input_tensor,
                class_index
            )

        st.subheader("🔮 Prediction")

        st.success(
            f"Predicted class: **{class_name}**"
        )

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("📊 Top 5 Predictions")

        st.dataframe(
            top5_predictions,
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "Confidence scores are model outputs, not guarantees "
            "that a prediction is correct."
        )

        st.divider()

        st.subheader("🔍 Grad-CAM Visual Explanation")

        st.write(
            "The heatmap highlights image regions that influenced "
            "the selected class score. Redder or brighter areas "
            "typically indicate stronger relative activation in "
            "this visualization."
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
            ax.imshow(
                cam,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1
            )

            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "Grad-CAM is an approximate visual explanation. "
            "It does not prove that the model used valid reasoning."
        )

    except Exception as error:
        st.error("Image analysis failed.")

        with st.expander("View technical error"):
            st.code(str(error))
