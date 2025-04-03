# YOLO Dataset Creator

A graphical tool for creating labeled datasets for YOLO (You Only Look Once) object detection models.

## Quick Start Guide

### Windows Users

1. **First time setup**: Double-click `install_dependencies.bat` to install all required dependencies.
2. **Run the application**: Double-click `run_yolo_creator.bat`

### Command Line (All Platforms)

1. **Install dependencies**:
   ```bash
   pip install -r yolo_dataset_creator/requirements.txt
   ```

2. **Run the application**:
   ```bash
   python yolo_dataset_creator.py
   ```

## Features

- Load multiple images
- Run a pre-trained YOLO model to get initial bounding box suggestions
- Manage class labels
- Assign correct class labels to detected bounding boxes
- Augment the dataset using Albumentations
- Save the annotated dataset in the standard YOLO format

## Detailed Documentation

For more detailed instructions and documentation, please see the [full README](yolo_dataset_creator/README.md). 