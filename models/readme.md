# Models Folder Overview

The `models` folder is intended to store various machine learning models, including YOLOv5 and YOLOv8, used in the ML-based suture pad analysis project. Although this folder is currently empty, users can clone the required models from the official Ultralytics repositories into this directory.

## Usage

This folder serves as a placeholder for managing YOLO models. It allows you to keep all model-related files in a centralized location for easy access and deployment in the project.

## Cloning YOLO Models

### 1. Cloning YOLOv5
To clone the YOLOv5 repository into this folder, use the following command:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

This will create a folder named yolov5 within the models directory. After cloning, you can use the YOLOv5 scripts for training, validation, and inference.

### 2. Cloning YOLOv8
To clone the YOLOv8 repository (Ultralytics), use the following command:
```bash
git clone https://github.com/ultralytics/ultralytics.git yolov8
```
This will create a folder named yolov8 within the models directory. The YOLOv8 repository contains all the necessary tools for model training, validation, and testing.

### Notes:
1. Ensure you have git installed on your system before running the above commands.
2. The cloned repositories will include prebuilt scripts for training, testing, and inference.
3. For additional dependencies, refer to the installation guides in the respective repositories.
