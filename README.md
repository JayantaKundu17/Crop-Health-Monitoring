Absolutely. Use this as your `README.md`. It is written to present the project as a serious ML portfolio project without overstating the real-world accuracy.

````markdown
# 🌱 Crop Health Monitoring System

AI-powered crop disease classification using **EfficientNet-B0**, with confidence scoring, Top-5 predictions, live camera input, visual severity estimation, and Grad-CAM explainability.

## 🚀 Live Demo

**Streamlit App:**  
https://crop-health-monitoring.streamlit.app/

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
````

---

## 📷 Live Camera Detection

The application includes Streamlit's camera input functionality.

Users can:

1. Open the web application.
2. Select **Take Photo**.
3. Capture an image of a crop leaf.
4. Submit the image.
5. Receive the predicted crop condition and confidence score.

The application can therefore be accessed from a device with a supported camera without requiring the user to run Python locally.

---

## 📈 Confidence Scores

For every input image, the model calculates class probabilities using Softmax.

The application displays:

* Most likely prediction
* Prediction confidence
* Top-5 predicted classes

Example:

```text
Detected Condition:
Tomato → Early Blight

Confidence:
96.42%

Top 5 Predictions:
1. Tomato → Early Blight       96.42%
2. Tomato → Late Blight         1.82%
3. Tomato → Target Spot         0.91%
4. Tomato → Leaf Mold           0.51%
5. Tomato → Septoria Leaf Spot   0.34%
```

---

## 🔍 Grad-CAM Explainability

The application optionally generates a **Grad-CAM visualization**.

Grad-CAM highlights image regions that contributed to the model's prediction.

This provides a visual explanation of where the model is focusing when making its classification.

```text
Original Leaf Image
        +
   EfficientNet-B0
        ↓
   Model Prediction
        ↓
     Grad-CAM
        ↓
Highlighted Regions
```

Grad-CAM is intended as an interpretability aid and should not be treated as a definitive biological explanation.

---

## 🩺 Visual Severity Estimate

The application also provides a simple visual severity estimate based on image color information.

The implementation uses:

* RGB → HSV conversion
* Green vegetation segmentation
* Approximation of yellow/brown damaged regions
* Percentage of affected pixels

The application categorizes the result as:

```text
< 10%       → LOW visual severity
10–30%      → MODERATE visual severity
> 30%       → HIGH visual severity
```

> **Important:** This is a simple color-based heuristic and is **not a clinically or agronomically validated disease-severity measurement**.

---

## 🗂️ Project Structure

```text
Crop-Health-Monitoring/
│
├── app.py
│
├── models/
│   └── crop_health_monitoring_model.pth
│
├── notebooks/
│   └── crop-health-monitoring.ipynb
│
├── results/
│   └── crop_health_metrics.csv
│
├── requirements.txt
│
└── .gitignore
```

### Files

**`app.py`**

Main Streamlit application containing:

* Model loading
* Image preprocessing
* Prediction
* Confidence calculation
* Top-5 predictions
* Grad-CAM
* Visual severity estimation
* Upload and camera interfaces

**`models/crop_health_monitoring_model.pth`**

Trained EfficientNet-B0 checkpoint containing the model weights, class information, image size, and evaluation metrics.

**`notebooks/crop-health-monitoring.ipynb`**

Training and evaluation notebook.

**`results/crop_health_metrics.csv`**

Evaluation metrics generated during model testing.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/JayantaKundu17/Crop-Health-Monitoring.git
```

Navigate to the project:

```bash
cd Crop-Health-Monitoring
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💻 Hardware Acceleration

The application automatically selects the available device in the following order:

```text
Apple MPS
    ↓
NVIDIA CUDA
    ↓
CPU
```

For example, on Apple Silicon Macs with MPS support:

```text
MPS available: True
```

the model can use Apple's Metal Performance Shaders backend for local inference.

---

## 📦 Technologies Used

### Machine Learning

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* ImageNet Transfer Learning

### Computer Vision

* OpenCV
* Pillow
* NumPy
* Grad-CAM

### Data & Evaluation

* Pandas
* Scikit-learn
* SciPy
* Matplotlib

### Deployment

* Streamlit
* GitHub

---

## 📚 Dataset

The model was trained and evaluated using the **PlantVillage dataset**.

PlantVillage provides labeled images of crop leaves covering multiple crops and disease categories.

The final model contains 38 classification categories.

---

## ⚠️ Limitations

This project is intended as an **AI-assisted crop disease classification prototype**.

Important limitations include:

* Performance on real-world field images may differ from the reported test performance.
* PlantVillage images may not fully represent field conditions.
* Lighting, camera quality, background, leaf orientation, and image quality can affect predictions.
* High model confidence does not guarantee a correct diagnosis.
* The visual severity calculation is a heuristic and is not agronomically validated.
* Predictions should not replace professional agricultural diagnosis.

---

## 🔮 Future Improvements

Potential improvements include:

* Training on more diverse field images
* Adding Indian crop-specific datasets
* Object detection for multiple leaves/plants in one image
* Improved disease severity estimation
* Weather and environmental data integration
* Treatment/recommendation system
* Mobile-friendly interface
* Model quantization for edge devices
* Deployment on mobile/IoT devices
* Continuous model improvement using real-world images

---

## 👨‍💻 Author

**Jayanta Kundu**

GitHub:
[https://github.com/JayantaKundu17](https://github.com/JayantaKundu17)

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.

````

### One thing I recommend before you paste it

Add **2–3 screenshots of your actual deployed application** near the top of the README. That will make the repository substantially stronger at first glance.

For example:

```markdown
## 🖥️ Application Preview

### Disease Classification

![Crop Health Monitoring](screenshots/prediction.png)

### Grad-CAM Explainability

![Grad-CAM](screenshots/gradcam.png)
````


