# Attention-Guided Knowledge Distillation for Lightweight and Explainable Pneumonia Classification

A deep learning project for pneumonia classification from chest X-ray images, focusing on **model compression, lightweight deployment, and interpretability** through knowledge distillation, attention transfer, and Grad-CAM.

## Project Overview

This project investigates whether knowledge from high-performing teacher models can be transferred to smaller neural networks while maintaining strong classification performance.

Multiple CNN architectures were trained and evaluated as teacher candidates and lightweight student models. Knowledge distillation was then combined with attention transfer to improve the performance of compact student networks.

Grad-CAM was also applied to visualize model attention and provide insight into the regions influencing predictions.

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle, containing pediatric chest X-ray images classified into two categories:

* **Normal**
* **Pneumonia**

The dataset is used for binary image classification.

## Methodology

The project follows the following workflow:

1. Preprocessed and augmented chest X-ray images.
2. Established a **Custom CNN baseline**.
3. Trained and evaluated multiple teacher candidate architectures.
4. Evaluated lightweight student architectures.
5. Applied **Knowledge Distillation (KD)** to transfer knowledge from teacher models to student models.
6. Incorporated **Attention Transfer (AT)** into the distillation process.
7. Evaluated models using classification and deployment-oriented metrics.
8. Used **Grad-CAM** for visual interpretation of model predictions.

## Models

### Baseline

* Custom CNN

### Teacher Candidates

* ResNet50
* EfficientNet-B0
* DenseNet121
* Three-teacher ensemble

### Student Models

* MobileNet
* TinyCNN

Both student architectures were evaluated in their plain, KD, and KD + Attention Transfer configurations.

## Results

Performance on the test partition:

| Model                 |   Accuracy |   F1-Score | Parameters |        Size | Inference Time |
| --------------------- | ---------: | ---------: | ---------: | ----------: | -------------: |
| Custom CNN            |     94.65% |     96.31% |      6.45M |    24.59 MB |      11.824 ms |
| ResNet50              |     98.18% |     98.76% |     23.51M |    89.69 MB |       9.261 ms |
| EfficientNet-B0       |     98.07% |     98.67% |      4.01M |    15.30 MB |       8.737 ms |
| DenseNet121           |     97.72% |     98.44% |      6.96M |    26.53 MB |       9.082 ms |
| 3-Teacher Ensemble    |     98.29% |     98.83% |          — |           — |              — |
| MobileNet Plain       |     98.41% |     98.91% |      2.23M |     8.49 MB |       8.977 ms |
| MobileNet KD          |     97.95% |     98.60% |      2.23M |     8.49 MB |       8.983 ms |
| **MobileNet KD + AT** | **98.29%** | **98.83%** |  **2.23M** | **8.49 MB** |   **8.964 ms** |
| TinyCNN Plain         |     94.88% |     96.47% |      1.61M |     6.15 MB |       8.711 ms |
| TinyCNN KD            |     95.11% |     96.64% |      1.61M |     6.15 MB |       8.765 ms |
| TinyCNN KD + AT       |     95.11% |     96.62% |      1.61M |     6.15 MB |       8.665 ms |

### Key Result

The **MobileNet KD + Attention Transfer** model achieved **98.29% test accuracy and 98.83% F1-score** while using only **2.23 million parameters** and **8.49 MB** of storage.

This provided a substantially smaller model than ResNet50 while achieving performance comparable to the three-teacher ensemble.

## Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to visualize regions of chest X-ray images receiving greater attention during classification.

This provides an interpretable view of model predictions and helps examine whether the models are focusing on clinically relevant image regions.

## Deployment Metrics

In addition to classification performance, models were evaluated using:

* Number of parameters
* Model storage size
* Inference time

These metrics were considered to assess the suitability of lightweight models for resource-constrained deployment.

## Technologies

* **Python**
* **PyTorch**
* **Convolutional Neural Networks (CNNs)**
* **Knowledge Distillation**
* **Attention Transfer**
* **Grad-CAM**
* **Kaggle**
* **Jupyter Notebook**

## Repository Contents

```text
attention-guided-knowledge-distillation/
│
├── attention-guided-knowledge-distillation.ipynb
└── README.md
```

The notebook contains the complete experimental workflow, including preprocessing, model training, evaluation, knowledge distillation, attention transfer, Grad-CAM visualizations, and deployment metric comparisons.

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle. The dataset contains chest X-ray images categorized into **Normal** and **Pneumonia** classes.

The dataset is not included in this repository. It can be accessed from the original Kaggle source:

[Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## Project Context

This project was developed as a machine learning project to explore **deep learning model evaluation, knowledge distillation, model compression, and explainable AI** for medical image classification.
