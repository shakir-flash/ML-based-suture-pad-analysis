
# ML-Based Suture Pad Analysis

This repository is designed for suture pad analysis using machine learning models such as YOLOv5 and YOLOv8. The project encompasses data preprocessing, model training, evaluation, and visualization tools.

---

## Repository Structure

### **Folders and Files**

#### **1. data/**
Contains all datasets, including raw, preprocessed, and labeled data used for training, validation, and testing.
- `raw_data/`: Unprocessed raw data.
- `preprocessed_images_all/`: Preprocessed images ready for training.
- `labeled_data/`: Labeled data for specific training tasks.
- `labeled_data_all/`: Comprehensive labeled datasets.
- `new_ignore/`: Final processed data for training and testing.

#### **2. models/**
Stores YOLOv5 and YOLOv8 model architectures and weights.

#### **3. notebooks/**
Contains Jupyter notebooks for exploratory data analysis and prototyping.

#### **4. results/**
Includes output metrics, confusion matrices, and images from model evaluation.
- `images/`: Example output images.
- `metrics/`: Training and validation performance metrics.

#### **5. runs/**
Stores results from training and validation runs, including model weights and predictions.

#### **6. scripts/**
Contains Python scripts and Jupyter notebooks for suture analysis workflows.
- `run_visualizer.py`: Visualizes predictions.
- `angle_manip.ipynb`: Performs advanced angular manipulations on predictions.
- YOLO scripts (`suture_obbyolo`, etc.) for running experiments.

#### **7. yolov5/**
Cloned YOLOv5 repository with additional training and testing scripts tailored to this project.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-name.git
cd ML-based-suture-pad-analysis
```

### 2. Create a Python Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r yolov5/requirements.txt
```

### 4. Install YOLOv5 and YOLOv8 Models
- Clone YOLOv5:
  ```bash
  git clone https://github.com/ultralytics/yolov5.git
  ```
- Clone YOLOv8:
  ```bash
  git clone https://github.com/ultralytics/ultralytics.git yolov8
  ```

---

## Usage

### **Training**
- Modify `data.yaml` for dataset paths.
- Train YOLOv5:
  ```bash
  python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
  ```
- Train YOLOv8:
  ```bash
  python yolov8/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov8n.pt
  ```

### **Inference**
Run inference on test images:
```bash
python yolov5/detect.py --source data/test/images --weights yolov5s.pt
```

### **Evaluate**
Evaluate the model on validation data:
```bash
python yolov5/val.py --data data.yaml --weights yolov5s.pt
```

---

## Dataset Details

### **Data Preparation**
- **Raw Data**: Stored in `data/raw_data/`.
- **Preprocessing**: Images undergo normalization, resizing, and augmentation.

### **Labeled Data**
- Format: YOLO format (text files with bounding boxes).
- Classes: `knot_good`, `knot_loose`, `suture_good`, `suture_loose`, etc.

---

## Key Features
- YOLOv5 and YOLOv8 integration.
- Automatic labeling tools.
- Advanced metrics visualization.

---
