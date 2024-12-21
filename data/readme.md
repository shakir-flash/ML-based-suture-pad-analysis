# Data Folder Overview

The `data` folder contains all the necessary datasets, preprocessed data, and related resources required for the ML-based suture pad analysis project. This folder is organized into subdirectories, each serving a specific purpose in the workflow of the project.

## Structure and Purpose of Subdirectories

### 1. **data_obb/**
Contains oriented bounding box (OBB) data used for advanced detection and analysis. It is primarily used for models requiring precise annotations.

### 2. **enhanced_images_all/**
Includes enhanced versions of images, processed to improve clarity and feature detection. These images are typically used in preprocessing and augmentation steps.

### 3. **isolated_sutures/**
Holds isolated suture images, extracted from raw data for focused analysis. This subdirectory is used for detailed examination of suture quality.

### 4. **labeled_data/**
Contains labeled data for training and testing. Each file includes annotations needed for model training.

### 5. **labeled_data_all/**
This is an extended version of the labeled data, combining multiple datasets into a single repository for comprehensive training purposes.

### 6. **new_ignore/**
Includes files or data points marked for exclusion from the training or evaluation process. These might be outliers or irrelevant data.

### 7. **preprocessed_images_all/**
Stores preprocessed images, formatted and cleaned for direct input into models. This ensures uniformity and compatibility across datasets.

### 8. **raw_data/**
Holds the original, unprocessed data directly obtained from sources. This serves as the starting point for all preprocessing and analysis tasks.

### 9. **.gitkeep**
A placeholder file to ensure empty directories are tracked in the version control system (Git).

## General Purpose of the `data` Folder
This folder acts as the backbone for dataset storage and organization. It streamlines the process of training and evaluating machine learning models by maintaining a clear structure for raw, labeled, and processed data. Each subdirectory has been designed to facilitate a specific step in the ML pipeline, from data collection to final analysis.

## Usage
- **Training Models**: Use `labeled_data/` or `labeled_data_all/` for labeled datasets.
- **Testing and Evaluation**: Utilize `isolated_sutures/` or `enhanced_images_all/` for specific model evaluations.
- **Preprocessing**: Modify or use data in `raw_data/` and store results in `preprocessed_images_all/`.

---
