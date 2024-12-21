# Train Folder Overview

The `train` folder contains outputs and artifacts generated during the training process of the detection model.

## Structure and Purpose

### 1. **args.yaml**
Configuration file used for training, detailing parameters and settings.

### 2. **confusion_matrix.png** and **confusion_matrix_normalized.png**
Visual representations of the confusion matrix, providing insights into model predictions.

### 3. **F1_curve.png, P_curve.png, PR_curve.png, and R_curve.png**
Graphs depicting performance metrics over training epochs.

### 4. **labels.jpg and labels_correlogram.jpg**
Visualizations of label distributions and correlations in the training data.

### 5. **train_batch0.jpg to train_batch272.jpg**
Batch visualizations of training samples, showing model inputs and predictions. These files provide insights into how the model processes and learns from the training data. Each batch file represents a snapshot of the data being used in a particular training iteration.

### 6. **val_batch0_labels.jpg and val_batch0_pred.jpg**
Validation batch visualizations, including ground truths and predictions.

### 7. **results.csv and results.png**
Aggregated results of training performance, stored as both a CSV file and a visualization.

### 8. **weights/**
Subfolder containing model weights (`best.pt`, `last.pt`) saved during training.

### 9. **events.out.tfevents.* (TensorBoard Log File)**
A log file generated during training, compatible with TensorBoard. This file contains detailed metrics such as loss, accuracy, and other performance indicators tracked during training. It is used for visualizing training progress and trends.

## Purpose

The `train` folder serves as a repository for all artifacts generated during training. It supports debugging, performance analysis, and model deployment.

## Usage
1. Use `args.yaml` to replicate training settings.
2. Analyze metrics and visualizations for insights into model performance.
3. Use batch visualizations (`train_batch0.jpg` to `train_batch272.jpg`) to understand how the model interacts with the training data.
4. Use `events.out.tfevents.*` with TensorBoard to visualize detailed training metrics.
5. Retrieve weights from the `weights/` subfolder for deployment or fine-tuning.

---
