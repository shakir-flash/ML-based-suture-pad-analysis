# Scripts Folder Overview

The `scripts` folder contains Python scripts, Jupyter Notebooks, and model weights required for the ML-based suture pad analysis project. These resources facilitate tasks such as data preprocessing, model training, evaluation, and visualization.

## Structure and Purpose of Files

### 1. **angle_manip.ipynb**
A Jupyter Notebook for manipulating and visualizing angles in suture datasets. Coded by Shakir Ahmed.

### 2. **run_visualizer.py**
A Python script for visualizing training and validation outputs. Coded by Shakir Ahmed

### 3. **SutureAnalysisUsingYOLO_Bible.ipynb**
The main Google Colab notebook containing comprehensive preprocessing, training, and postprocessing steps. This includes:
- Grayscale, contour, and edge detection preprocessing.
- YOLOv8 training with hyperparameters, upgrading from earlier tests in YOLOv5.
- Postprocessing steps to calculate metrics such as suture length, knot-to-incision distance, and tail length.
Additionally, this notebook contains preparatory code for cropping original dataset images with scales for future tasks.
Coded by Surajit Pal.

### 4. **SutureAnalysisUsingYOLO_Ref.ipynb**
A concise Google Colab notebook focusing on the preprocessing, contour detection training and postprocessing, which was the best-performing method. It includes detection outputs used for the iShowcase poster. Coded by Surajit Pal.

### 5. **SutureAnalysisUsingTensorFlow_ShakirRef.ipynb**
A Google Colab notebook derived from `SutureAnalysisUsingYOLO_Ref.ipynb`, provided for merging tasks in collaboration. It is simplified for external sharing and integration. Coded by Surajit Pal and merged with Shakir Ahmed.

### 6. **sutureNew.ipynb**
A general-purpose Google Colab notebook for exploratory analysis and suture data processing. Coded by Shakir Ahmed.

### 7. **yolov8n.pt**
YOLOv8 model weights used for training or inference.

### 8. **suture_obbyolo/**
Subfolder containing scripts for YOLO-based object detection tasks.

### 9. **.gitkeep**
A placeholder file used to ensure that empty directories are tracked by Git. This file has no content but is essential for version control.

### 10. **.gitattributes**
A configuration file used by Git to define how certain files are handled in the repository. This may include line-ending normalization or specific handling of binary files.

## Purpose of the `scripts` Folder

The `scripts` folder centralizes project-specific code, enabling streamlined workflows for training, evaluation, and analysis.

## Usage

1. Navigate to the relevant file or subfolder for specific tasks.
2. Modify the scripts or notebooks to adapt to your dataset and requirements.

---
