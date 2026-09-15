# 🌱 Crop Health Monitoring System

AI-powered crop disease classification using **EfficientNet-B0**, with confidence scoring, Top-5 predictions, live camera input, visual severity estimation, and Grad-CAM explainability.

## 🚀 Live Demo

**Streamlit App:**  
https://crop-health-monitoring.streamlit.app/

---

##SCREENSHOTS


<img width="1461" height="786" alt="Screenshot 2026-09-15 at 5 29 54 AM" src="https://github.com/user-attachments/assets/5f8ca1d6-5a19-48e6-85e4-e16713312750" />

<img width="1461" height="750" alt="Screenshot 2026-09-15 at 5 30 01 AM" src="https://github.com/user-attachments/assets/f4c09dd1-e632-433c-a21d-a21a3e957f48" />

<img width="1455" height="786" alt="Screenshot 2026-09-15 at 5 30 28 AM" src="https://github.com/user-attachments/assets/839c245f-ae4a-49f1-9fa1-0a923146a805" />

<img width="1439" height="781" alt="Screenshot 2026-09-15 at 5 34 14 AM" src="https://github.com/user-attachments/assets/972049e1-9a9e-4cfe-9b8d-110461df7d84" />

<img width="1451" height="802" alt="Screenshot 2026-09-15 at 5 33 58 AM" src="https://github.com/user-attachments/assets/dfd46ae8-7554-4561-803c-117f9fd9fdd7" />




---
## 📌 Overview

Crop diseases can significantly affect agricultural productivity when they are not identified early.

This project uses a deep learning image classification model to identify crop diseases from leaf images. The system supports both uploaded images and live camera input through a Streamlit web application.

The trained model classifies images into **38 crop-health/disease categories**.

### Key Features

- Deep learning-based crop disease classification
- EfficientNet-B0 architecture
- ImageNet transfer learning
- 38 crop/disease classes
- Image upload support
- Live camera input
- Prediction confidence scores
- Top-5 predictions
- Grad-CAM model explainability
- Color-based visual severity estimation
- GPU acceleration when available
- Streamlit web interface

---

## 🧠 Model

The classification model is based on **EfficientNet-B0** with ImageNet transfer learning.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Input Size | 224 × 224 |
| Number of Classes | 38 |
| Transfer Learning | ImageNet |
| Framework | PyTorch |
| Deployment | Streamlit |

The final classification layer was modified to output predictions for all 38 classes.

---

## 📊 Performance

The model was evaluated on a held-out test set from the PlantVillage dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | **99.54%** |
| Precision | **99.55%** |
| Recall | **99.54%** |
| F1 Score | **99.54%** |
| Training Images | 43,444 |
| Test Images | 5,431 |
| Number of Classes | 38 |

> **Note:** These metrics represent performance on the held-out PlantVillage test set. Real-world performance may differ, particularly for photographs taken under different lighting, backgrounds, camera conditions, or field environments.

---

## 🌾 Supported Classes

The model supports the following 38 classes:

### Apple
- Apple — Apple Scab
- Apple — Black Rot
- Apple — Cedar Apple Rust
- Apple — Healthy

### Blueberry
- Blueberry — Healthy

### Cherry
- Cherry — Powdery Mildew
- Cherry — Healthy

### Corn
- Corn — Cercospora Leaf Spot / Gray Leaf Spot
- Corn — Common Rust
- Corn — Northern Leaf Blight
- Corn — Healthy

### Grape
- Grape — Black Rot
- Grape — Esca / Black Measles
- Grape — Leaf Blight
- Grape — Healthy

### Orange
- Orange — Huanglongbing / Citrus Greening

### Peach
- Peach — Bacterial Spot
- Peach — Healthy

### Pepper
- Pepper — Bacterial Spot
- Pepper — Healthy

### Potato
- Potato — Early Blight
- Potato — Late Blight
- Potato — Healthy

### Raspberry
- Raspberry — Healthy

### Soybean
- Soybean — Healthy

### Squash
- Squash — Powdery Mildew

### Strawberry
- Strawberry — Leaf Scorch
- Strawberry — Healthy

### Tomato
- Tomato — Bacterial Spot
- Tomato — Early Blight
- Tomato — Late Blight
- Tomato — Leaf Mold
- Tomato — Septoria Leaf Spot
- Tomato — Spider Mites
- Tomato — Target Spot
- Tomato — Tomato Yellow Leaf Curl Virus
- Tomato — Tomato Mosaic Virus
- Tomato — Healthy

---

## 🔬 How It Works

```text
                Input Image
                     │
             ┌───────┴────────┐
             │                │
        Upload Image      Take Photo
             │                │
             └───────┬────────┘
                     │
                     ▼
              Image Preprocessing
                 224 × 224
                     │
                     ▼
              EfficientNet-B0
                     │
                     ▼
             Softmax Prediction
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Disease    Confidence   Top-5
      Prediction     Score    Predictions
          │
          ├───────────────┐
          │               │
          ▼               ▼
   Visual Severity     Grad-CAM
      Estimate       Explainability






