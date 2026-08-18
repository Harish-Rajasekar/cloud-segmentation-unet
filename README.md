# ☁️ Cloud Segmentation using U-Net

A deep learning project for **multi-class semantic segmentation of clouds in satellite imagery** using a U-Net architecture with a pretrained ResNet34 encoder.

The project explores the complete computer vision workflow — from exploratory data analysis and RLE mask decoding to model training, evaluation, and qualitative prediction analysis.

---

## 📌 Project Overview

Satellite images can contain multiple types of cloud formations with different shapes, sizes, and visual characteristics. Accurately identifying these regions at the pixel level is a semantic segmentation problem.

This project builds a deep learning pipeline to segment four cloud categories:

* **Fish**
* **Flower**
* **Gravel**
* **Sugar**

The model predicts a separate segmentation mask for each cloud category.

---

## 🎯 Objectives

* Explore and understand the satellite image dataset.
* Decode Run-Length Encoded (RLE) segmentation masks.
* Build a multi-class semantic segmentation pipeline using PyTorch.
* Train a U-Net model with a pretrained ResNet34 encoder.
* Improve the baseline using higher-resolution inputs and data augmentation.
* Evaluate performance using Dice Score and IoU.
* Analyze segmentation performance visually and on a per-class basis.

---

## 🧠 Model Architecture

The project uses **U-Net with a pretrained ResNet34 encoder**.

```text
Satellite Image
      │
      ▼
Data Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
ResNet34 Encoder
      │
      ▼
U-Net Decoder
      │
      ▼
4-Channel Segmentation Output
      │
      ▼
Fish / Flower / Gravel / Sugar
```

The ResNet34 encoder extracts hierarchical visual features while the U-Net decoder reconstructs spatial information required for pixel-level segmentation.

---

## 🔄 Project Pipeline

```text
EDA
 │
 ▼
RLE Mask Decoding
 │
 ▼
Train / Validation / Test Split
 │
 ▼
Image Preprocessing
 │
 ▼
Data Augmentation
 │
 ▼
U-Net + ResNet34
 │
 ▼
BCE + Dice Loss
 │
 ▼
Model Training
 │
 ▼
Validation & Learning Rate Scheduling
 │
 ▼
Test Set Evaluation
 │
 ▼
Prediction Visualization
```

---

## 🗂️ Dataset

The dataset contains satellite imagery with segmentation annotations represented using **Run-Length Encoding (RLE)**.

Each image can contain multiple cloud categories, with the segmentation masks represented separately for each class.

The four target classes are:

| Class  | Description                  |
| ------ | ---------------------------- |
| Fish   | Fish-type cloud formations   |
| Flower | Flower-type cloud formations |
| Gravel | Gravel-type cloud formations |
| Sugar  | Sugar-type cloud formations  |

---

## 🧪 Experiments

Two model configurations were developed.

### Version 1 — Baseline

The initial model established a baseline for segmentation performance.

### Version 2 — Enhanced Pipeline

The second version introduced:

* **512 × 512 input resolution**
* Stronger image augmentation
* Learning-rate scheduling
* Early stopping
* U-Net with ResNet34 encoder
* BCE + Dice loss

The purpose of Version 2 was to investigate whether higher-resolution inputs and a stronger training pipeline could improve segmentation performance.

---

## 📊 Results

The final model was evaluated on a held-out test set.

| Metric         |    Score |
| -------------- | -------: |
| **Dice Score** | **0.59** |
| **IoU**        | **0.42** |

### Per-Class Dice

| Cloud Class | Dice Score |
| ----------- | ---------: |
| Fish        |       0.51 |
| Flower      |       0.59 |
| Gravel      |       0.52 |
| Sugar       |   **0.61** |

Sugar achieved the highest Dice score, while Fish was comparatively more challenging.

---

## 📈 Training Analysis

The training curves show that the model learned rapidly during the initial epochs and gradually converged.

Training and validation loss decreased during the main training period, while validation Dice improved and eventually stabilized. Towards the later epochs, validation loss increased and Dice decreased, indicating the onset of overfitting.

Early stopping was therefore used to prevent unnecessary training after validation performance stopped improving.

---

## 🖼️ Prediction Results

The test-set predictions demonstrate that the model successfully identifies major cloud regions while occasionally producing additional predictions or overestimating cloud boundaries.

The qualitative results are consistent with the overall Dice and IoU scores.

The notebook contains detailed prediction visualizations comparing:

**Original Image → Ground Truth → Model Prediction**

---

## 📚 Notebooks

### `01_EDA.ipynb`

Exploratory analysis of:

* Dataset structure
* Image dimensions
* Cloud class distribution
* RLE annotations
* Class presence
* Sample satellite imagery

### `02_Model_Training.ipynb`

Complete deep learning pipeline covering:

* RLE mask decoding
* Dataset preparation
* Data augmentation
* PyTorch Dataset and DataLoader
* U-Net architecture
* ResNet34 encoder
* BCE + Dice loss
* Model training
* Validation
* Test evaluation
* Prediction visualization
* Training curves
* Per-class analysis
* Version comparison

---

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* Albumentations
* Segmentation Models PyTorch
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Scikit-learn

---

## 📁 Repository Structure

```text
cloud-segmentation-unet/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/<your-username>/cloud-segmentation-unet.git
cd cloud-segmentation-unet
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The notebooks can then be opened using Jupyter Notebook, JupyterLab, or VS Code.

> **Note:** The original dataset is not included in this repository. Dataset paths in the notebooks may need to be adjusted before running them locally.

---

## 🔮 Future Improvements

Potential directions for improving the model include:

* Stronger encoder architectures
* Transformer-based segmentation models
* Test-Time Augmentation (TTA)
* Mixed-precision training
* Ensemble methods
* Improved post-processing
* Hyperparameter optimization

---

## 👤 Author

**Harish R**

Electrical & Electronics Engineering
Aspiring AI/ML Engineer
