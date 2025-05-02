# YOLO Dataset Creator

A graphical application for creating labeled datasets specifically formatted for training YOLO (You Only Look Once) object detection models.

## Features

- Load multiple images from your computer
- Run a pre-trained YOLO model to get initial bounding box suggestions
- Manage class labels
- Assign correct class labels to detected bounding boxes
- Augment the dataset using Albumentations
- Save the annotated dataset in the standard YOLO format

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository or download the source code:

```bash
git clone https://github.com/yourusername/yolo-dataset-creator.git
cd yolo-dataset-creator
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:

```bash
pip install -r yolo_dataset_creator/requirements.txt
```

Note: Installing PyTorch might require specific commands depending on your system/CUDA version. If you encounter issues, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions.

## Running the Application

You have several ways to run the application:

### 1. Using the batch file (Windows)

Simply double-click on the `run_yolo_creator.bat` file.

### 2. Using the Python launcher script

```bash
python yolo_dataset_creator.py
```

### 3. Using the main module directly

```bash
python -m yolo_dataset_creator.main
```

### 4. Install as a package (advanced)

Install the package in development mode:

```bash
pip install -e .
```

Then run using the console entry point:

```bash
yolo-dataset-creator
```

## Usage

Once the application is running:

1. Click **"Add Images..."** and select the image files you want to include.

2. Click **"Select Model..."** and choose a compatible YOLO model file (e.g., `.pt` or `.onnx`).

3. Select an image from the **Image List** on the left.

4. Click **"Process Selected Image"** to run detection. Wait for bounding boxes to appear.

5. Define necessary classes using the **Class List** controls on the right ("Add Class").

6. Assign classes:
   - Click on a bounding box in the **Image Annotation Area** to select it (it will be highlighted).
   - Click the correct class name in the **Class List**. The label on the box will update.
   - Repeat for all boxes in all images.

7. (Optional) Configure the number of **Augmentations per image**.

8. (Optional) Click **"Augment Dataset"**. This will process *all* annotated images.

9. Click **"Save Dataset..."**. Choose an output directory where the `dataset/` folder will be created.

10. The dataset (images, labels, train/val splits, YAML file) will be saved in the chosen location, ready to be used for training a YOLO model.

## Output Format

The saved dataset will have the following structure:

```
dataset/
├── images/                 # Contains all original and augmented images
├── labels/                 # Contains corresponding .txt annotation files
├── train.txt               # List of image paths for training
├── val.txt                 # List of image paths for validation
└── data.yaml               # Configuration file for training
```

Each label file (`.txt`) contains one line per object in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```
where all values are normalized to be between 0 and 1.

## Troubleshooting

If you encounter any import errors when running the application, try:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r yolo_dataset_creator/requirements.txt
   ```

2. Make sure you're running the application from the root directory of the project.

3. If using a virtual environment, ensure it's activated.

## License

[MIT License](LICENSE) 