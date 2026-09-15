import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/crop_health_monitoring_model.pth"

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# IMAGE TRANSFORM
# EXACTLY MATCHES NOTEBOOK EVALUATION TRANSFORM
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, class_names, class_to_idx


model, class_names, class_to_idx = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, top_k=5):

    image = image.convert("RGB")

    tensor = eval_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        values, indices = torch.topk(
            probabilities,
            k=min(top_k, len(class_names)),
            dim=1
        )

    results = []

    for probability, index in zip(
        values[0].detach().cpu().numpy(),
        indices[0].detach().cpu().numpy()
    ):

        results.append({
            "class": class_names[int(index)],
            "confidence": float(probability)
        })

    return results


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image):

    image = image.convert("RGB")

    rgb = np.array(
        image.resize((IMG_SIZE, IMG_SIZE))
    ).astype(np.float32) / 255.0

    input_tensor = eval_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        logits = model(input_tensor)

        predicted_class = (
            logits.argmax(dim=1).item()
        )

    target_layer = model.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


# ============================================================
# VISUAL SEVERITY ESTIMATE
# SAME HEURISTIC FROM NOTEBOOK
# ============================================================

def estimate_visual_severity(image):

    image = np.array(
        image.convert("RGB")
    )

    # RGB → HSV
    import cv2

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    # Approximate green vegetation
    lower_green = np.array(
        [25, 30, 20]
    )

    upper_green = np.array(
        [100, 255, 255]
    )

    leaf_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Approximate yellow/brown damaged pixels
    lower_damage = np.array(
        [5, 30, 20]
    )

    upper_damage = np.array(
        [35, 255, 255]
    )

    damage_mask = cv2.inRange(
        hsv,
        lower_damage,
        upper_damage
    )

    leaf_pixels = np.count_nonzero(
        leaf_mask
    )

    if leaf_pixels == 0:
        return None

    damaged_pixels = np.count_nonzero(
        damage_mask & leaf_mask
    )

    severity = (
        damaged_pixels /
        leaf_pixels
    ) * 100

    return float(
        np.clip(
            severity,
            0,
            100
        )
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Health Monitoring",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Crop Health Monitoring")

st.markdown(
    """
    **AI-powered crop disease classification using EfficientNet-B0**

    Upload an image or take a picture using your camera.
    The model predicts the most likely crop disease and provides
    confidence scores and an optional Grad-CAM explanation.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Architecture:** EfficientNet-B0"
    )

    st.write(
        "**Input:** 224 × 224"
    )

    st.write(
        "**Classes:** 38"
    )

    st.write(
        "**Training:** ImageNet transfer learning"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.divider()

    st.caption(
        "Model trained on the PlantVillage dataset."
    )

    st.caption(
        "The reported 99.54% test accuracy "
        "is based on the held-out PlantVillage test set."
    )


# ============================================================
# INPUT
# ============================================================

st.subheader("1. Provide a crop image")

input_method = st.radio(
    "Choose input method:",
    [
        "Upload Image",
        "Take Photo"
    ],
    horizontal=True
)

image = None

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a leaf/crop image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

else:

    camera_file = st.camera_input(
        "Take a picture of the crop leaf"
    )

    if camera_file is not None:

        image = Image.open(
            camera_file
        ).convert("RGB")


# ============================================================
# RUN PREDICTION
# ============================================================

if image is not None:

    st.divider()

    st.subheader("2. Image")

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.image(
            image,
            caption="Input image",
            width="stretch"
        )

    with st.spinner(
        "Analyzing crop image..."
    ):

        predictions = predict_image(
            image,
            top_k=5
        )

    top_prediction = predictions[0]

    predicted_class = (
        top_prediction["class"]
    )

    confidence = (
        top_prediction["confidence"]
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    with col2:

        st.subheader(
            "Prediction"
        )

        st.metric(
            "Detected Condition",
            predicted_class.replace(
                "___",
                " → "
            ).replace(
                "_",
                " "
            )
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        if confidence >= 0.90:

            st.success(
                "High-confidence prediction"
            )

        elif confidence >= 0.70:

            st.warning(
                "Moderate-confidence prediction"
            )

        else:

            st.error(
                "Low-confidence prediction"
            )


    # ========================================================
    # TOP 5
    # ========================================================

    st.divider()

    st.subheader(
        "Top 5 Predictions"
    )

    prediction_df = pd.DataFrame({
        "Prediction": [
            p["class"].replace(
                "___",
                " → "
            ).replace(
                "_",
                " "
            )
            for p in predictions
        ],
        "Confidence": [
            p["confidence"]
            for p in predictions
        ]
    })

    prediction_df["Confidence"] = (
        prediction_df["Confidence"]
        .map(
            lambda x: f"{x:.2%}"
        )
    )

    st.dataframe(
        prediction_df,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # VISUAL SEVERITY
    # ========================================================

    st.divider()

    st.subheader(
        "Visual Severity Estimate"
    )

    severity = estimate_visual_severity(
        image
    )

    if severity is None:

        st.info(
            "Severity estimate unavailable "
            "because a clear vegetation region "
            "could not be detected."
        )

    else:

        st.metric(
            "Estimated affected vegetation",
            f"{severity:.2f}%"
        )

        if severity < 10:

            st.success(
                "LOW visual severity"
            )

        elif severity < 30:

            st.warning(
                "MODERATE visual severity"
            )

        else:

            st.error(
                "HIGH visual severity"
            )

        st.caption(
            "This is a simple color-based visual "
            "estimate and is not a clinically or "
            "agronomically validated disease-severity measurement."
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader(
        "Model Explainability — Grad-CAM"
    )

    show_gradcam = st.checkbox(
        "Generate Grad-CAM visualization"
    )

    if show_gradcam:

        with st.spinner(
            "Generating Grad-CAM..."
        ):

            try:

                cam_image = generate_gradcam(
                    image
                )

                st.image(
                    cam_image,
                    caption=(
                        "Grad-CAM — regions "
                        "influencing the prediction"
                    ),
                    width="stretch"
                )

            except Exception as e:

                st.error(
                    f"Grad-CAM could not be generated: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Crop Health Monitoring | EfficientNet-B0 | "
    "PlantVillage 38-class classification"
)
