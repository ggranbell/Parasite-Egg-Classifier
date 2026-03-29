# Multi-Class Parasite Egg Classification Using ConvNeXt with Grad-CAM Explainability

---

## Table of Contents

1. Introduction
2. Dataset Description
3. Data Preprocessing and Augmentation
4. Model Architecture
5. Training Configuration
6. Training Process and Convergence
7. Evaluation Results
8. Grad-CAM Explainability Analysis
9. System Architecture and Deployment
10. Hardware and Software Environment
11. Conclusions and Future Work

---

## 1. Introduction

This document presents the complete technical documentation of a multi-class image classification system designed to identify intestinal parasite eggs from microscopy images. The system classifies images into three species:

- **Ascaris lumbricoides**
- **Hookworm**
- **Trichuris trichiura**

These three species are among the most prevalent soil-transmitted helminths (STH) worldwide, collectively infecting over 1.5 billion people according to the World Health Organization. Accurate and automated identification of parasite eggs from stool microscopy samples can significantly assist clinical diagnostics, particularly in resource-limited settings where trained parasitologists may not be readily available.

The system uses **ConvNeXt-Base**, a modern convolutional neural network architecture pretrained on ImageNet-1K, adapted for this domain through transfer learning. To enhance interpretability, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is integrated to visualize which regions of the microscopy image the model focuses on when making its classification decisions.

---

## 2. Dataset Description

### 2.1 Overview

The dataset consists of microscopy images of parasite eggs organized into three classes. Images are in JPEG format and stored in class-labeled subdirectories.

### 2.2 Class Distribution

| Class | Label Index | Description |
|---|---|---|
| Ascaris lumbricoides | 0 | Large roundworm; eggs are oval with a thick, irregular outer shell |
| Hookworm | 1 | Necator/Ancylostoma species; eggs are oval, thin-shelled, and transparent |
| Trichuris trichiura | 2 | Whipworm; eggs are barrel-shaped with bipolar plugs |

### 2.3 Dataset Split

The dataset is divided into three splits for training, validation, and testing. Each split is balanced across all three classes.

| Split | Ascaris lumbricoides | Hookworm | Trichuris trichiura | Total |
|---|---|---|---|---|
| Training | 140 | 140 | 140 | 420 |
| Validation | 20 | 20 | 20 | 60 |
| Test | 40 | 40 | 40 | 120 |
| **Total** | **200** | **200** | **200** | **600** |

The split ratio is approximately **70% / 10% / 20%** (train / validation / test). All three classes are perfectly balanced within each split, which eliminates class imbalance as a confounding factor.

### 2.4 Manifest Files

Each split is defined by a manifest text file stored in the `data/` directory:

- `data/train_set.txt` — 420 entries
- `data/val_set.txt` — 60 entries
- `data/test_set.txt` — 120 entries

Each line contains a relative path without the file extension. The class label is derived from the directory structure: `dataset/<split>/<class_name>/<filename>`.

---

## 3. Data Preprocessing and Augmentation

### 3.1 Preprocessing Pipeline

All images undergo the following deterministic preprocessing before being fed into the model:

1. **Resize**: Images are resized to 416 × 416 pixels (image_size + 32 margin).
2. **Center Crop**: A center crop of 384 × 384 pixels is applied to remove border artifacts.
3. **Tensor Conversion**: Images are converted to PyTorch tensors with pixel values in [0, 1].
4. **Normalization**: ImageNet normalization statistics are applied:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### 3.2 Data Augmentation (Training Only)

To improve generalization on the small training set, the following constrained augmentations are applied to training images only. Validation and test sets receive no augmentation.

**Shearing (p = 0.7)**
Two severity categories are defined. One is selected at random each time the augmentation fires:
- Category 1 (Mild): Shear angle randomly sampled from ±15°
- Category 2 (Strong): Shear angle randomly sampled from ±30°

**Rotation (p = 0.7)**
Three categorical rotation angles are defined. One is selected at random:
- Category 1: 90°
- Category 2: 180°
- Category 3: 270°

Each augmentation is applied independently with a probability of 0.7 per image. The augmentation strategy is intentionally constrained to shearing and rotation because parasite eggs can appear at arbitrary orientations under the microscope, but their morphological shape and internal structure should not be distorted by color jittering or elastic transformations.

### 3.3 Summary of Transform Pipelines

**Training Pipeline:**
```
Resize(416) → CenterCrop(384) → RandomShear(±15° or ±30°, p=0.7) → RandomRotation(90°/180°/270°, p=0.7) → ToTensor → Normalize
```

**Validation / Test Pipeline:**
```
Resize(416) → CenterCrop(384) → ToTensor → Normalize
```

---

## 4. Model Architecture

### 4.1 Backbone: ConvNeXt-Base

The classification model is built upon **ConvNeXt-Base**, a modernized pure-CNN architecture proposed by Liu et al. (2022). The model is loaded from the `timm` (PyTorch Image Models) library with weights pretrained on ImageNet-1K (1.28 million images, 1000 classes).

**Key properties of the backbone:**

| Property | Value |
|---|---|
| Architecture | ConvNeXt-Base |
| Pretraining Dataset | ImageNet-1K |
| Input Resolution | 384 × 384 pixels |
| Backbone Output Dimensionality | 1024 |
| Total Parameters | ~88.6 million |

### 4.2 Classification Head

The original ImageNet classification head (1000-class linear layer) is removed and replaced with a custom 3-class head designed for this task:

```
ConvNeXt-Base Backbone (pretrained, all layers trainable)
        ↓
    [1024-dim feature vector]
        ↓
    LayerNorm(1024)
        ↓
    Dropout(p = 0.3)
        ↓
    Linear(1024 → 3)
        ↓
    Logits [batch_size × 3]
```

**Design rationale:**
- **LayerNorm** before the classifier stabilizes the feature distribution from the backbone.
- **Dropout (p = 0.3)** regularizes the classifier to prevent overfitting on the small 420-image training set.
- The **entire backbone is kept trainable** (all layers fine-tuned), as the domain gap between ImageNet natural images and microscopy images is significant.

### 4.3 Why ConvNeXt?

ConvNeXt was selected over alternatives (ResNet, EfficientNet, Vision Transformers) for the following reasons:

1. **Modern pure-CNN architecture**: No attention layers means more efficient GPU utilization and simpler deployment compared to ViT-based models.
2. **Strong pretrained features**: ImageNet pretraining provides rich low-level feature representations (edges, textures) that transfer well to microscopy image analysis.
3. **VRAM efficiency**: ConvNeXt-Base at 384 × 384 input resolution fits comfortably within 16 GB VRAM with mixed-precision training enabled.
4. **Competitive accuracy**: Matches or exceeds Vision Transformers on equivalent parameter budgets while being faster to train.

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Learning Rate | 1 × 10⁻⁴ | Initial learning rate for AdamW optimizer |
| Batch Size | 16 | Maximum that fits in 16 GB VRAM at 384 × 384 with AMP |
| Maximum Epochs | 50 | Upper bound; early stopping may terminate earlier |
| Weight Decay | 1 × 10⁻² | L2 regularization via AdamW |
| Label Smoothing | 0.1 | Softens one-hot targets to reduce overconfidence |
| Dropout | 0.3 | Applied in the classification head |
| Gradient Clipping | max_norm = 1.0 | Prevents exploding gradients |
| Early Stopping Patience | 10 | Training stops if validation loss does not improve for 10 consecutive epochs |
| Random Seed | 42 | Fixed for reproducibility |

### 5.2 Optimizer

**AdamW** (Adam with decoupled weight decay) is used as the optimizer. AdamW correctly applies weight decay as a direct penalty on the weight magnitudes, rather than through the gradient like the original Adam, making it more effective for regularization in transfer learning settings.

### 5.3 Learning Rate Schedule

**Cosine Annealing** is used to smoothly decay the learning rate from the initial value (1 × 10⁻⁴) to a minimum value (η_min = 1 × 10⁻⁶) over the course of training. This schedule provides:
- Fast initial learning when the loss landscape has large gradients
- Fine-grained updates near convergence when the model is close to a local minimum

### 5.4 Mixed Precision Training (AMP)

Automatic Mixed Precision (AMP) with FP16 is enabled throughout training. This provides:
- **~2× throughput** by leveraging the GPU's Tensor Cores for FP16 computation
- **~50% reduction in VRAM usage**, allowing larger batch sizes or higher resolution inputs
- Maintained numerical stability through the GradScaler which dynamically adjusts loss scaling

### 5.5 Loss Function

**Cross-Entropy Loss** with label smoothing (ε = 0.1) is used. With label smoothing, the target distribution becomes:

```
y_smooth = (1 - ε) × y_one_hot + ε / K
```

where K = 3 (number of classes). This prevents the model from becoming overly confident in its predictions and improves generalization.

---

## 6. Training Process and Convergence

### 6.1 Training Summary

The model was trained for **23 epochs** before early stopping was triggered (patience = 10 epochs). The best model checkpoint was saved at the epoch achieving the highest validation accuracy.

### 6.2 Epoch-by-Epoch Training History

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) | Learning Rate |
|---|---|---|---|---|---|
| 1 | 0.6718 | 0.4291 | 77.64 | 93.33 | 1.00 × 10⁻⁴ |
| 2 | 0.3583 | 0.3188 | 98.08 | 100.00 | 9.99 × 10⁻⁵ |
| 3 | 0.3320 | 0.4061 | 99.28 | 91.67 | 9.96 × 10⁻⁵ |
| 4 | 0.3088 | 0.3069 | 100.00 | 100.00 | 9.91 × 10⁻⁵ |
| 5 | 0.3098 | 0.3166 | 99.52 | 98.33 | 9.84 × 10⁻⁵ |
| 6 | 0.3038 | 0.3305 | 100.00 | 98.33 | 9.76 × 10⁻⁵ |
| 7 | 0.3025 | 0.2994 | 99.76 | 100.00 | 9.65 × 10⁻⁵ |
| 8 | 0.3015 | 0.3144 | 100.00 | 98.33 | 9.53 × 10⁻⁵ |
| 9 | 0.3009 | 0.3063 | 100.00 | 100.00 | 9.39 × 10⁻⁵ |
| 10 | 0.2994 | 0.3268 | 100.00 | 96.67 | 9.23 × 10⁻⁵ |
| 11 | 0.2995 | 0.3021 | 99.76 | 100.00 | 9.05 × 10⁻⁵ |
| 12 | 0.2976 | 0.2948 | 100.00 | 100.00 | 8.86 × 10⁻⁵ |
| 13 | 0.2968 | 0.2935 | 100.00 | 100.00 | 8.66 × 10⁻⁵ |
| 14 | 0.2959 | 0.2945 | 100.00 | 100.00 | 8.44 × 10⁻⁵ |
| 15 | 0.2957 | 0.2945 | 100.00 | 100.00 | 8.21 × 10⁻⁵ |
| 16 | 0.2954 | 0.3003 | 100.00 | 100.00 | 7.96 × 10⁻⁵ |
| 17 | 0.2956 | 0.3013 | 100.00 | 98.33 | 7.70 × 10⁻⁵ |
| 18 | 0.2948 | 0.2996 | 100.00 | 100.00 | 7.43 × 10⁻⁵ |
| 19 | 0.2945 | 0.3071 | 100.00 | 98.33 | 7.16 × 10⁻⁵ |
| 20 | 0.2942 | 0.3081 | 100.00 | 98.33 | 6.87 × 10⁻⁵ |
| 21 | 0.2942 | 0.3050 | 100.00 | 98.33 | 6.58 × 10⁻⁵ |
| 22 | 0.2942 | 0.3116 | 100.00 | 98.33 | 6.28 × 10⁻⁵ |
| 23 | 0.2938 | 0.3163 | 100.00 | 98.33 | 5.98 × 10⁻⁵ |

### 6.3 Convergence Analysis

**Training loss** converged rapidly from 0.6718 (epoch 1) to approximately 0.294 by epoch 10, then plateaued. The fast initial convergence is attributed to the strong ImageNet pretrained features.

**Training accuracy** reached 100% by epoch 4 and remained at or near 100% for the remainder of training, indicating that the model fully memorized the training set. This is expected given the relatively small dataset size (420 images).

**Validation accuracy** achieved 100% multiple times (epochs 2, 4, 7, 9, 11–16, 18) and oscillated between 98.33% and 100% thereafter. The model's best validation loss was recorded at epoch 13 (val_loss = 0.2935).

**Learning rate** followed the cosine annealing schedule, decaying smoothly from 1 × 10⁻⁴ to 5.98 × 10⁻⁵ over 23 epochs.

**Early stopping** was triggered at epoch 23 because the validation loss did not improve beyond the epoch 13 minimum for 10 consecutive epochs, indicating convergence.

---

## 7. Evaluation Results

### 7.1 Test Set Performance

The best model checkpoint was evaluated on the held-out test set (120 images: 40 per class). No augmentation was applied during evaluation.

**Overall Test Accuracy: 100.00%**

The model correctly classified all 120 test images across all three classes.

### 7.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Ascaris lumbricoides | 1.0000 | 1.0000 | 1.0000 | 40 |
| Hookworm | 1.0000 | 1.0000 | 1.0000 | 40 |
| Trichuris trichiura | 1.0000 | 1.0000 | 1.0000 | 40 |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | **120** |

All precision, recall, and F1-score metrics are 1.0000, indicating perfect classification on the test set.

### 7.3 Confusion Matrix

The confusion matrix confirms zero misclassifications across all class pairs:

|  | Predicted: Ascaris | Predicted: Hookworm | Predicted: Trichuris |
|---|---|---|---|
| **Actual: Ascaris** | **40** | 0 | 0 |
| **Actual: Hookworm** | 0 | **40** | 0 |
| **Actual: Trichuris** | 0 | 0 | **40** |

All values lie perfectly on the diagonal, with zero off-diagonal entries (no false positives or false negatives for any class).

### 7.4 Per-Image Predictions

All 120 individual test predictions were verified correct:
- Images 0–39: Ascaris lumbricoides → correctly predicted as Ascaris lumbricoides
- Images 40–79: Hookworm → correctly predicted as Hookworm
- Images 80–119: Trichuris trichiura → correctly predicted as Trichuris trichiura

---

## 8. Grad-CAM Explainability Analysis

### 8.1 Overview

Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize which regions of the input image contributed most to the model's classification decisions. This technique generates a heatmap by computing the gradients of the predicted class score with respect to the feature maps of the last convolutional layer in the backbone.

### 8.2 Implementation

The Grad-CAM implementation targets the **last block of the final stage** of the ConvNeXt backbone (`model.backbone.stages[-1].blocks[-1]`). For each input image:

1. A forward pass is performed to obtain the class prediction.
2. Gradients of the predicted class logit are backpropagated to the target layer.
3. The feature map activations are weighted by their average gradient (channel-wise global average pooling).
4. The weighted activations are summed and passed through a ReLU to produce the heatmap.
5. The heatmap is resized to the input image dimensions and overlaid using the JET colormap.

### 8.3 Results

Grad-CAM visualizations were generated for all 120 test images (40 per class) and saved to `outputs/cams/`. The heatmaps demonstrate that the model consistently focuses on:

- **Ascaris lumbricoides**: The model attends to the distinctive thick, mammillated outer shell and the internal cellular structure of the egg.
- **Hookworm**: Attention is concentrated on the thin, transparent shell and the morula (cell mass) within the egg.
- **Trichuris trichiura**: The model focuses on the characteristic barrel shape and the bipolar mucous plugs at each end.

These attention patterns confirm that the model has learned biologically meaningful features rather than relying on background artifacts or staining differences, supporting the clinical validity of the classification decisions.

### 8.4 Web-Based Visualization

A Flask-based web application provides real-time Grad-CAM visualization through an interactive user interface. Users can upload single or multiple microscopy images and receive:
- The predicted class and confidence score
- Side-by-side comparison of the original image and the Grad-CAM overlay
- A probability distribution chart showing confidence across all three classes

---

## 9. System Architecture and Deployment

### 9.1 Project Structure

```
Thesis-MultiClass-Image-Classification/
├── data/                   # Dataset manifests (train/val/test lists)
│   ├── train_set.txt
│   ├── val_set.txt
│   └── test_set.txt
├── src/                    # Core library modules
│   ├── config.py           # Centralized hyperparameters and paths
│   ├── dataset.py          # Custom Dataset and DataLoader factory
│   ├── model.py            # ConvNeXt classifier architecture
│   ├── utils.py            # Utilities (EarlyStopping, plotting, seeding)
│   └── visualize_cam.py    # Grad-CAM visualization engine
├── scripts/                # Command-line tools
│   ├── train.py            # GPU-accelerated training loop
│   ├── evaluate.py         # Test set evaluation and confusion matrix
│   ├── analysis.py         # Academic analysis (ROC, t-SNE, cost)
│   ├── inference.py        # CLI inference on unseen images
│   ├── tune.py             # Phased hyperparameter tuning
│   └── compare_results.py  # Cross-phase tuning comparison
├── templates/
│   └── index.html          # Web UI frontend
├── app.py                  # Flask web application backend
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Docker service configuration
└── requirements.txt        # Python dependencies
```

### 9.2 Web Application

The system includes a Flask-based web application that provides:

- **Single Image Upload**: Upload one microscopy image for immediate classification.
- **Batch Processing**: Upload multiple images simultaneously, processed sequentially with a progress indicator.
- **Results Gallery**: Thumbnail grid view of all processed images with their predictions.
- **Detailed View**: For each image, a side-by-side comparison of the original image and the Grad-CAM explainability overlay, along with a class probability distribution.

### 9.3 Docker Deployment

The application is containerized for easy deployment:

```bash
# Build and start the container
docker-compose up --build

# Access at http://localhost:5000
```

The Docker image uses `python:3.11-slim` as the base, installs CPU-only PyTorch for portability, and mounts the `outputs/` directory to access the trained model checkpoint.

---

## 10. Hardware and Software Environment

### 10.1 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| CPU | AMD Ryzen 7 7700 (8 cores / 16 threads) |
| RAM | 32 GB DDR5 |
| Storage | NVMe SSD |
| Operating System | Windows |

### 10.2 Software Dependencies

| Package | Purpose |
|---|---|
| Python | 3.11 |
| PyTorch | Deep learning framework |
| torchvision | Image transforms and pretrained model utilities |
| timm | PyTorch Image Models (ConvNeXt backbone loader) |
| Pillow | Image loading and manipulation |
| NumPy | Numerical computation |
| Matplotlib | Training curve and analysis plot generation |
| Seaborn | Confusion matrix heatmap visualization |
| scikit-learn | Classification report, ROC curves, t-SNE |
| tqdm | Progress bar for training loops |
| Flask | Web application framework |
| OpenCV | Grad-CAM heatmap overlay processing |
| Gunicorn | Production WSGI server |

### 10.3 GPU Optimizations

| Optimization | Benefit |
|---|---|
| Mixed Precision Training (AMP FP16) | ~2× throughput, ~50% VRAM reduction |
| pin_memory = True | Faster CPU → GPU data transfer |
| num_workers = 6 | Saturates the data loading pipeline (8-core CPU, 2 reserved for OS) |
| Gradient Clipping (max_norm = 1.0) | Prevents gradient explosion during fine-tuning |

---

## 11. Conclusions and Future Work

### 11.1 Summary of Results

The ConvNeXt-Base model achieved **100% accuracy** on the held-out test set of 120 microscopy images (40 per class), with perfect precision, recall, and F1-score across all three parasite species. Grad-CAM analysis confirmed that the model focuses on biologically relevant morphological features of the parasite eggs, rather than artifact-based shortcuts.

### 11.2 Key Findings

1. **Transfer learning is highly effective** for this domain: Strong ImageNet pretraining enabled rapid convergence (100% train accuracy by epoch 4) even with a small dataset of 420 training images.
2. **Constrained augmentation is sufficient**: Simple shearing and rotation augmentations, designed to reflect natural variation in microscope slide orientation, proved adequate without aggressive augmentations.
3. **Regularization prevented overfitting**: The combination of dropout (0.3), label smoothing (0.1), and weight decay (0.01) successfully regularized the model despite the small dataset size.
4. **Grad-CAM validates clinical relevance**: The model's attention patterns align with the morphological features that parasitologists use for manual identification, supporting the system's trustworthiness for clinical decision support.

### 11.3 Limitations

1. **Small dataset size**: While the model achieved perfect accuracy, the test set contains only 120 images. Larger and more diverse datasets would provide more robust performance estimates.
2. **Limited class diversity**: The system currently classifies only three species. Expanding to additional parasite species would increase clinical utility.
3. **Single imaging condition**: The dataset represents a specific microscopy protocol. Performance may vary with different staining techniques, magnification levels, or microscope equipment.

### 11.4 Future Work

1. **Expand the dataset** with additional microscopy images from multiple clinical sites to improve generalizability.
2. **Add more parasite species** to extend the classifier to a broader range of soil-transmitted helminths.
3. **Cross-validation and multi-seed evaluation** for rigorous statistical significance testing.
4. **Deployment to mobile devices** for point-of-care diagnostics in resource-limited settings.
5. **Integration with digital microscopy** pipelines for fully automated slide analysis.

---

## Appendix A: Training Curves

The training curves (loss and accuracy per epoch) are saved to `outputs/training_curves.png`. The plots show:
- **Loss Curve**: Training loss converges rapidly and stabilizes around 0.294. Validation loss follows a similar trend with slight oscillation.
- **Accuracy Curve**: Training accuracy reaches 100% by epoch 4. Validation accuracy oscillates between 98.33% and 100% before early stopping.

---

## Appendix B: Hyperparameter Tuning Strategy

An 8-phase greedy grid search strategy is available for systematic hyperparameter optimization:

| Phase | Parameters Tuned | Trials | Estimated Time |
|---|---|---|---|
| 1 | Learning Rate | 5 | ~3.75 hours |
| 2 | Batch Size + Weight Decay | 6 | ~4.5 hours |
| 3 | Dropout + Label Smoothing | 5 | ~3.75 hours |
| 4 | Model Variant + Resolution | 3 | ~2.25 hours |
| 5 | LR Scheduler (Warmup + η_min) | 4 | ~3 hours |
| 6 | Loss Function (Focal Loss + Class Weights) | 5 | ~3.75 hours |
| 7 | Differential LR (Backbone Multiplier) | 3 | ~2.25 hours |
| 8 | Augmentation Probability | 4 | ~3 hours |
| **Total** | | **35** | **~26.25 hours** |

Each phase runs full 50-epoch training with early stopping. The best hyperparameters from each phase are locked in before the subsequent phase begins, ensuring an efficient sequential search.

---

## Appendix C: Computational Cost

| Metric | Value |
|---|---|
| Model | ConvNeXt-Base |
| Input Resolution | 384 × 384 |
| Total Parameters | ~88.6 million |
| Trainable Parameters | ~88.6 million |
| FLOPs | ~45.2 GFLOPs |
| Checkpoint Size | ~1.05 GB |

---

*Document generated on March 29, 2026.*
*Project: Thesis-MultiClass-ImageClassification*
