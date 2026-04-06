<div align="center">
  <h1>🔬 Multi-Class Parasite Egg Classification</h1>
  <p>
    <b>An academic-grade computer vision system for automated identification of intestinal parasite eggs from microscopy images.</b>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/Status-Research--Ready-2ECC71?style=for-the-badge" alt="Status" />
  </p>
</div>

---

## 📖 Overview

**Multi-Class Parasite Egg Classification** is a  deep learning project that leverages the **ConvNeXt-Base** architecture with transfer learning to achieve high-accuracy classification of soil-transmitted helminth (STH) eggs from microscopy images. The system ships with a full scientific pipeline — from dataset preparation and model training, to evaluation, explainability (Grad-CAM), and a deployable Flask web interface — making it ready for both academic review and practical clinical screening use.

## ✨ Features

- **🏆 State-of-the-Art Accuracy**: Achieves **100% Accuracy** on a balanced test set across three parasite species using a fine-tuned ConvNeXt-Base backbone.
- **💡 Explainable AI (XAI)**: Integrated Grad-CAM visualization highlights the morphological features (shells, plugs, internal masses) the model relies on for each prediction.
- **🌐 Interactive Web UI**: A modern, dark-themed Flask interface for drag-and-drop image upload, real-time batch classification, probability charts, and Grad-CAM heatmap display.
- **📊 Academic Analysis Suite**: Built-in scripts for multi-seed statistical reporting, ROC/AUC curve generation, t-SNE feature embedding, and computational cost profiling.
- **🐳 Deployment Ready**: Fully containerized with Docker and Docker Compose for consistent, reproducible cross-platform performance.
- **🛡️ Reproducible Science**: Fixed random seeds, stratified data splits, and manifest-based data loading ensure every experiment is fully reproducible.

## 🧬 Target Species

The system classifies three of the most prevalent intestinal parasites worldwide:

| Species | Morphology |
|---------|------------|
| **Ascaris lumbricoides** | Large roundworm eggs with thick, mammillated outer shells |
| **Hookworm** | Necator/Ancylostoma eggs featuring thin, transparent shells |
| **Trichuris trichiura** | Whipworm eggs with characteristic barrel shape and bipolar plugs |

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch](https://pytorch.org/) + [timm](https://github.com/huggingface/pytorch-image-models) (ConvNeXt-Base with ImageNet pretrained weights).
- **Data & Evaluation**: NumPy, Pandas, scikit-learn (classification reports, confusion matrices).
- **Visualization**: Matplotlib, Seaborn (training curves, heatmaps, ROC plots).
- **Explainability**: OpenCV + custom Grad-CAM implementation for class activation mapping.
- **Web Application**: [Flask](https://flask.palletsprojects.com/) with Gunicorn for production serving.
- **Containerization**: [Docker](https://www.docker.com/) + Docker Compose.

## 🚀 Quick Setup

### Prerequisites

1. **Python**: v3.11 or higher.
2. **CUDA GPU**: Recommended for training (tested on RTX 5060 Ti 16GB).

### Installation

1. **Clone and Install**

```bash
git clone https://github.com/yourusername/Thesis-MultiClass-ImageClassification.git
cd Thesis-MultiClass-ImageClassification

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install PyTorch with CUDA**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. **Run the Web UI**

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser to access the interactive classification dashboard.

### 🐳 Using Docker

```bash
docker-compose up --build
```

## 🧪 Scientific Workflow

### 1. Training

```bash
# Standard training run (50 epochs, early stopping)
python scripts/train.py

# Debug / smoke test (CPU, 1 epoch, 2 batches)
python scripts/train.py --debug

# Multi-seed run for statistical reporting
python scripts/train.py --seeds 42,123,456,789,1234
```

### 2. Evaluation

```bash
python scripts/evaluate.py
```

### 3. Academic Analysis (ROC, t-SNE, Cost)

```bash
python scripts/analysis.py --all
```

All outputs are saved to the `outputs/analysis/` directory.

## 📂 Project Architecture

```text
Thesis-MultiClass-ImageClassification/
├── data/                 # Dataset manifests (train/val/test splits)
│   ├── train_set.txt     # 420 training images
│   ├── val_set.txt       # 60 validation images
│   └── test_set.txt      # 120 test images
├── src/                  # Core Logic
│   ├── config.py         # Hyperparameters and configuration
│   ├── dataset.py        # PyTorch Dataset and DataLoader
│   ├── model.py          # ConvNeXt architecture and custom head
│   ├── utils.py          # Helper functions and metrics
│   └── visualize_cam.py  # Grad-CAM implementation
├── scripts/              # CLI Tools
│   ├── train.py          # Training with AMP, cosine annealing, early stopping
│   ├── evaluate.py       # Test set evaluation and confusion matrix
│   ├── tune.py           # Phased hyperparameter grid search
│   ├── analysis.py       # ROC/AUC, t-SNE, computational cost
│   └── inference.py      # Single/batch image inference
├── templates/            # Flask Web UI (HTML/CSS/JS)
├── outputs/              # Checkpoints, training curves, and plots
├── app.py                # Flask Web Application
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container orchestration
└── DOCUMENTATION.md      # Complete technical reference
```

## 📖 Documentation

Detailed technical documentation for each phase is available in the [`docs/`](docs/) directory:

- [**Overview**](docs/00_overview.md) — Project summary and quick start
- [**Dataset Setup**](docs/01_dataset_setup.md) — Data organization and manifest generation
- [**Model Architecture**](docs/02_model_architecture.md) — ConvNeXt backbone and classification head
- [**Training**](docs/03_training.md) — Training loop, optimizer, and scheduler details
- [**Evaluation**](docs/04_evaluation.md) — Metrics, confusion matrix, and result analysis
- [**Hyperparameter Tuning**](docs/05_hyperparameter_tuning.md) — Phased grid search strategy
- [**Inference**](docs/06_inference.md) — CLI and Web UI usage
- [**Academic Analysis**](docs/07_academic_analysis.md) — ROC, t-SNE, cost, and multi-seed reporting

## ⚙️ Hardware Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| **CPU** | AMD Ryzen 7 7700 |
| **RAM** | 32GB DDR5 |

