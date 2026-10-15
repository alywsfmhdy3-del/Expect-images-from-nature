# Natural Scene Image Classification (VGG16 + SVM)

A hybrid computer vision pipeline that classifies natural scenes into six categories by combining deep transfer learning (**VGG16**) for feature extraction with a **Support Vector Machine (SVM)** for classification, achieving **91% test accuracy** across 3,000 evaluation images.

---

## 📌 Project Overview

Instead of training a deep convolutional neural network from scratch, this project leverages a pre-trained **VGG16** model (trained on ImageNet) as a fixed feature extractor. The extracted 512-dimensional feature representations are standardized and fed into an **RBF-kernel SVM** classifier (`C=50`, `gamma=0.001`) to achieve high classification performance with rapid training times.

### Target Classes
The model classifies images into 6 distinct scene categories:
* 🏔️ `mountain`
* 🛣️ `street`
* 🧊 `glacier`
* 🏢 `buildings`
* 🌊 `sea`
* 🌲 `forest`

---

## ⚙️ Pipeline Architecture

1. **Image Preprocessing:**
   * Resizes input images to `224 × 224` pixels.
   * Converts images to NumPy arrays and applies VGG16-specific preprocessing (`preprocess_input`).
2. **Deep Feature Extraction (VGG16):**
   * Uses `VGG16(weights='imagenet', include_top=False, pooling='avg')`.
   * Removes the top classification layers and applies Global Average Pooling to output a compact **512-dimensional feature vector** per image.
3. **Data Shuffling & Scaling:**
   * Shuffles training samples (`random_state=42`) to prevent order bias.
   * Normalizes features to zero mean and unit variance using `StandardScaler`.
4. **Classification (SVM):**
   * Trains a Support Vector Classifier (`SVC`) with a Radial Basis Function (`rbf`) kernel (`C=50`, `gamma=0.001`).

---

## 📊 Experimental Results

Evaluated on **3,000 test images**, the hybrid VGG16 + SVM pipeline achieved an overall accuracy of **91.0%**.

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **mountain** | 0.86 | 0.86 | 0.86 | 525 |
| **street** | 0.93 | 0.92 | 0.93 | 501 |
| **glacier** | 0.85 | 0.83 | 0.84 | 553 |
| **buildings** | 0.91 | 0.90 | 0.91 | 437 |
| **sea** | 0.92 | 0.96 | 0.94 | 510 |
| **forest** | **0.99** | **0.98** | **0.99** | 474 |
| | | | | |
| **Accuracy** | | | **0.91** | **3000** |
| **Macro Avg** | **0.91** | **0.91** | **0.91** | **3000** |
| **Weighted Avg** | **0.91** | **0.91** | **0.91** | **3000** |

### Key Findings
* **Best Performing Class:** `forest` achieved a near-perfect **0.99 F1-score**, followed closely by `sea` (**0.94**) and `street` (**0.93**), due to their highly distinctive textures and structural patterns.
* **Primary Error Source:** `glacier` (**0.84 F1-score**) and `mountain` (**0.86 F1-score**) show the lowest scores, as snow-covered mountains and glaciers share overlapping visual features in both color and geometry.

---

## 🛠️ Tech Stack & Requirements

* **Python** 3.8+
* **TensorFlow / Keras** (`tensorflow`)
* **Scikit-Learn** (`scikit-learn`)
* **NumPy** (`numpy`)
* **OpenCV** (`opencv-python`)
* **Matplotlib** (`matplotlib`)
* **tqdm** (`tqdm`)

Install all required dependencies via pip:

```bash
pip install numpy opencv-python tqdm scikit-learn tensorflow matplotlib

```

📂 Dataset Structure
The script expects the dataset to be organized in the following directory structure:
```bash
dataset/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_test/
    └── seg_test/
        ├── buildings/
        ├── forest/
        ├── glacier/
        ├── mountain/
        ├── sea/
        └── street/

```
🚀 Usage
Update the dataset paths inside load_data_vgg() if running outside the Kaggle environment:

```bash

datasets_paths = [
    r'path/to/seg_train',
    r'path/to/seg_test'
]
Run the script to extract features, train the SVM classifier, and plot the confusion matrix:

python main.py