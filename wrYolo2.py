import sys
import os
import subprocess
import threading
import re # For parsing epoch progress
from pathlib import Path
import json # For potential future config saving

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
    QMessageBox, QGroupBox, QProgressBar, QFormLayout, QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon # Optional: for setting an icon

# --- Configuration & Constants ---
APP_NAME = "YOLOv8 Trainer GUI"
APP_VERSION = "1.1.0"
# Find a suitable icon file (e.g., .ico or .png) if you have one
# APP_ICON_PATH = "path/to/your/icon.png"

DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 8 # Adjusted for wider compatibility
DEFAULT_PROJECT_DIR = "runs/train"
DEFAULT_EXPERIMENT_NAME = "yolo_experiment"
DEFAULT_LR0 = 0.01
DEFAULT_PATIENCE = 50
DEFAULT_WEIGHT_DECAY = 0.0005

DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 9pt; /* Slightly smaller for more density */
}
QGroupBox {
    background-color: #313131; /* Slightly lighter than main bg */
    border: 1px solid #454545;
    border-radius: 5px;
    margin-top: 1ex;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center; /* Center title */
    padding: 0 5px;
    color: #00aaff; /* Accent color for title */
}
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3c3c3c;
    color: #f0f0f0;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px;
    min-height: 20px; /* Ensure consistent height */
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0078d7;
}
QTextEdit {
    selection-background-color: #0078d7;
    selection-color: #ffffff;
}
QPushButton {
    background-color: #4a4a4a;
    color: #f0f0f0;
    border: 1px solid #5f5f5f;
    border-radius: 3px;
    padding: 6px 12px;
    min-width: 90px;
}
QPushButton:hover {
    background-color: #5a5a5a;
    border: 1px solid #6f6f6f;
}
QPushButton:pressed {
    background-color: #404040;
}
QPushButton:disabled {
    background-color: #383838;
    color: #707070;
}
#StartButton {
    background-color: #4CAF50; /* Green */
    font-weight: bold;
}
#StartButton:hover { background-color: #5cb85c; }
#StartButton:disabled { background-color: #38753a; color: #a0a0a0; }

#StopButton {
    background-color: #f44336; /* Red */
    font-weight: bold;
}
#StopButton:hover { background-color: #e57373; }
#StopButton:disabled { background-color: #a33029; color: #a0a0a0; }

QLabel {
    color: #f0f0f0;
    padding-top: 3px; /* Align better with QFormLayout inputs */
}
QProgressBar {
    border: 1px solid #555555;
    border-radius: 3px;
    text-align: center;
    color: #f0f0f0;
    background-color: #3c3c3c;
}
QProgressBar::chunk {
    background-color: #0078d7;
    border-radius: 2px;
    margin: 0.5px;
}
QScrollBar:vertical {
    border: 1px solid #454545; background: #3c3c3c; width: 12px; margin: 12px 0 12px 0; border-radius: 3px;
}
QScrollBar::handle:vertical { background: #666666; min-height: 20px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 10px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal {
    border: 1px solid #454545; background: #3c3c3c; height: 12px; margin: 0px 12px 0 12px; border-radius: 3px;
}
QScrollBar::handle:horizontal { background: #666666; min-width: 20px; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 10px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
QMessageBox { background-color: #3c3c3c; }
QMessageBox QLabel { color: #f0f0f0; }
QMessageBox QPushButton { background-color: #4a4a4a; color: #f0f0f0; border: 1px solid #5f5f5f; padding: 5px; min-width: 70px; }
QMessageBox QPushButton:hover { background-color: #5a5a5a; }
QComboBox QAbstractItemView {
    background-color: #3c3c3c; color: #f0f0f0; border: 1px solid #555555;
    selection-background-color: #0078d7; selection-color: #ffffff; padding: 2px;
}
QStatusBar { background-color: #2b2b2b; color: #f0f0f0; }
QStatusBar::item { border: none; } /* Remove border for items in status bar */
"""

# --- Worker Thread for Training ---
class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)
    progress_bar_signal = pyqtSignal(int)

    def __init__(self, cmd_args, total_epochs):
        super().__init__()
        self.cmd_args = cmd_args
        self.total_epochs = total_epochs if total_epochs > 0 else 1 # Avoid division by zero
        self.process = None
        self._is_running = True
        self.epoch_pattern = re.compile(rf"^\s*(\d+)/{self.total_epochs}\s+") # Regex for "  1/100 "

    def run(self):
        self.progress_signal.emit(f"Starting training (Total Epochs: {self.total_epochs})...\n")
        self.progress_signal.emit(f"Command: {' '.join(self.cmd_args)}\n---\n")
        self._is_running = True
        exit_code = -1
        final_message = "An unexpected error occurred during training setup."

        try:
            self.process = subprocess.Popen(
                self.cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                shell=False
            )

            current_epoch_for_bar = 0
            while self._is_running:
                if self.process.stdout is None: break
                line = self.process.stdout.readline()
                if not line:
                    break
                self.progress_signal.emit(line)

                match = self.epoch_pattern.search(line)
                if match:
                    try:
                        epoch_current = int(match.group(1))
                        if epoch_current > current_epoch_for_bar : # Update only on new epoch
                            current_epoch_for_bar = epoch_current
                            progress_percent = int(((epoch_current -1) / self.total_epochs) * 100) # 0-indexed for bar
                            self.progress_bar_signal.emit(progress_percent)
                    except ValueError:
                        pass # Ignore if parsing fails for a specific line

            if self.process: # Check if process was initialized
                self.process.wait()
                exit_code = self.process.returncode

            if not self._is_running and exit_code != 0:
                final_message = "Training stopped by user."
                self.progress_signal.emit("\n---\nTraining process stopped manually.\n")
            elif exit_code == 0:
                final_message = "Training completed successfully!"
                self.progress_bar_signal.emit(100)
                self.progress_signal.emit("\n---\nTraining finished successfully.\n")
            else:
                final_message = f"Training failed with exit code: {exit_code}"
                self.progress_signal.emit(f"\n---\nTraining process failed (Exit Code: {exit_code}).\n")

        except FileNotFoundError:
            final_message = "Error: 'yolo' (or python -m ultralytics.cli) command not found. Is Ultralytics installed and in PATH/Python environment?"
            self.progress_signal.emit(f"\nCritical Error: {final_message}\n")
            exit_code = -2
        except Exception as e:
            final_message = f"An error occurred during training execution: {e}"
            self.progress_signal.emit(f"\n---\nAn error occurred during training:\n{e}\n")
            exit_code = -3
        finally:
            self._is_running = False
            if self.process: # Ensure process is cleaned up
                if self.process.poll() is None: # if still running
                    try:
                        self.process.kill() # Force kill if not stopped properly
                        self.process.wait(timeout=2)
                    except Exception as e_kill:
                        self.progress_signal.emit(f"Error during final process kill: {e_kill}\n")
                self.process = None
            self.finished_signal.emit(exit_code, final_message)

    def stop(self):
        if self.process and self._is_running:
            self.progress_signal.emit("\n---\nAttempting to stop training...\n")
            self._is_running = False # Signal loop to exit
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.progress_signal.emit("Process did not terminate gracefully, forcing kill.\n")
                    self.process.kill()
                    self.process.wait(timeout=2) # Wait for kill
            except Exception as e:
                self.progress_signal.emit(f"Error while trying to stop process: {e}\n")
        else:
            self._is_running = False # Ensure flag is set

# --- Main Application Window ---
class YoloTrainerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.init_ui()
        self.check_ultralytics_installation()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        # if os.path.exists(APP_ICON_PATH):
        #     self.setWindowIcon(QIcon(APP_ICON_PATH))

        main_layout = QVBoxLayout(self)

        # --- Dataset & Model Configuration ---
        dataset_model_group = QGroupBox("Dataset & Model Configuration")
        dataset_model_form_layout = QFormLayout()

        # YAML Path
        self.yaml_path_edit = QLineEdit()
        self.yaml_path_edit.setPlaceholderText("Path to your dataset.yaml file")
        self.yaml_browse_button = QPushButton("Browse...")
        self.yaml_browse_button.clicked.connect(self.browse_yaml)
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(self.yaml_path_edit)
        yaml_layout.addWidget(self.yaml_browse_button)
        dataset_model_form_layout.addRow("Dataset YAML:", yaml_layout)

        # Base Model Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt", # Segmentation models
            "yolov8n-pose.pt", "yolov8s-pose.pt", # Pose models
            # Add more or custom ones as needed
        ])
        self.model_combo.setToolTip("Select a pre-trained YOLO model or specify a custom one below.")
        dataset_model_form_layout.addRow("Base Model:", self.model_combo)

        # Custom Model Path
        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("(Optional) Path to custom .pt model file")
        self.custom_model_browse_button = QPushButton("Browse...")
        self.custom_model_browse_button.clicked.connect(self.browse_custom_model)
        custom_model_layout = QHBoxLayout()
        custom_model_layout.addWidget(self.custom_model_path_edit)
        custom_model_layout.addWidget(self.custom_model_browse_button)
        dataset_model_form_layout.addRow("Custom Model Path:", custom_model_layout)

        dataset_model_group.setLayout(dataset_model_form_layout)
        main_layout.addWidget(dataset_model_group)


        # --- Output Configuration ---
        output_config_group = QGroupBox("Output Configuration")
        output_config_form_layout = QFormLayout()

        self.project_dir_edit = QLineEdit(DEFAULT_PROJECT_DIR)
        self.project_dir_edit.setToolTip("Parent directory for all training runs (YOLO 'project' arg).")
        self.project_dir_browse_button = QPushButton("Browse...")
        self.project_dir_browse_button.clicked.connect(self.browse_project_dir)
        project_dir_layout = QHBoxLayout()
        project_dir_layout.addWidget(self.project_dir_edit)
        project_dir_layout.addWidget(self.project_dir_browse_button)
        output_config_form_layout.addRow("Project Directory:", project_dir_layout)

        self.experiment_name_edit = QLineEdit(DEFAULT_EXPERIMENT_NAME)
        self.experiment_name_edit.setToolTip("Name for this specific training run folder (YOLO 'name' arg).")
        output_config_form_layout.addRow("Experiment Name:", self.experiment_name_edit)

        output_config_group.setLayout(output_config_form_layout)
        main_layout.addWidget(output_config_group)


        # --- Training Parameters ---
        params_group = QGroupBox("Core Training Parameters")
        params_form_layout = QFormLayout()

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(DEFAULT_EPOCHS)
        self.epochs_spinbox.setToolTip("Number of training epochs.")
        params_form_layout.addRow("Epochs:", self.epochs_spinbox)

        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 8192)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(DEFAULT_IMG_SIZE)
        self.imgsz_spinbox.setToolTip("Target image size (pixels, must be multiple of 32).")
        params_form_layout.addRow("Image Size (px):", self.imgsz_spinbox)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(-1, 1024) # -1 for AutoBatch
        self.batch_spinbox.setValue(DEFAULT_BATCH_SIZE)
        self.batch_spinbox.setToolTip("Images per batch (-1 for AutoBatch). Adjust based on GPU VRAM.")
        params_form_layout.addRow("Batch Size:", self.batch_spinbox)

        self.device_combo = QComboBox()
        self.device_combo.addItems(self.get_available_devices())
        self.device_combo.setToolTip("Select compute device. GPU recommended.")
        params_form_layout.addRow("Device:", self.device_combo)

        params_group.setLayout(params_form_layout)
        main_layout.addWidget(params_group)

        # --- Hyperparameters ---
        hyper_params_group = QGroupBox("Hyperparameters")
        hyper_params_form_layout = QFormLayout()

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "Adam", "SGD", "auto"])
        self.optimizer_combo.setCurrentText("auto")
        self.optimizer_combo.setToolTip("Choose the optimizer. 'auto' lets YOLO decide.")
        hyper_params_form_layout.addRow("Optimizer:", self.optimizer_combo)

        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setRange(0.00001, 1.0)
        self.lr0_spinbox.setSingleStep(0.0001)
        self.lr0_spinbox.setDecimals(5)
        self.lr0_spinbox.setValue(DEFAULT_LR0)
        self.lr0_spinbox.setToolTip("Initial learning rate.")
        hyper_params_form_layout.addRow("Learning Rate (lr0):", self.lr0_spinbox)

        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(0, 1000) # 0 means no early stopping
        self.patience_spinbox.setValue(DEFAULT_PATIENCE)
        self.patience_spinbox.setToolTip("Epochs to wait for improvement before early stopping (0 to disable).")
        hyper_params_form_layout.addRow("Patience:", self.patience_spinbox)

        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setRange(0.0, 0.1)
        self.weight_decay_spinbox.setSingleStep(0.00005)
        self.weight_decay_spinbox.setDecimals(5)
        self.weight_decay_spinbox.setValue(DEFAULT_WEIGHT_DECAY)
        self.weight_decay_spinbox.setToolTip("Optimizer weight decay.")
        hyper_params_form_layout.addRow("Weight Decay:", self.weight_decay_spinbox)

        hyper_params_group.setLayout(hyper_params_form_layout)
        main_layout.addWidget(hyper_params_group)


        # --- Output Console ---
        output_group = QGroupBox("Training Output")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Training output will appear here...")
        self.output_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group, 1) # Make output_group stretchable

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Epoch Progress: %p%")
        main_layout.addWidget(self.progress_bar)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.setObjectName("StartButton") # For specific styling
        self.start_button.clicked.connect(self.start_training)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setObjectName("StopButton") # For specific styling
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready.")

        self.setLayout(main_layout)
        self.resize(800, 750)


    def get_available_devices(self):
        """Detects available compute devices (CPU, CUDA, MPS)."""
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    devices.append(f"cuda:{i} ({device_name})")
                if torch.cuda.device_count() == 0: # Should be caught by is_available, but just in case
                    devices.append("cuda:0") # Offer default if detection fails
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices.append("mps") # For Apple Silicon
        except ImportError:
            self.append_output("Warning: PyTorch not found. CUDA/MPS device detection skipped. Offering defaults.")
            devices.append("cuda:0")
        except Exception as e:
            self.append_output(f"Warning: Error detecting Torch devices: {e}. Offering defaults.")
            devices.append("cuda:0")
        return devices

    def check_ultralytics_installation(self):
        """Checks if ultralytics seems to be installed."""
        try:
            process_check = subprocess.run(
                [sys.executable, "-m", "ultralytics.cli", "help"],
                capture_output=True, text=True, check=False, timeout=5
            )
            if process_check.returncode != 0 or "ultralytics" not in process_check.stdout.lower():
                QMessageBox.warning(
                    self, "Ultralytics Check Failed",
                    "Could not verify Ultralytics installation. 'python -m ultralytics.cli help' failed or did not produce expected output.\n"
                    "Please ensure Ultralytics is installed correctly in your Python environment.\n"
                    f"Attempted command output (stderr):\n{process_check.stderr}"
                )
                self.status_bar.showMessage("Warning: Ultralytics not found or not working.", 10000)
            else:
                self.status_bar.showMessage("Ultralytics installation check passed.", 5000)
        except FileNotFoundError:
            QMessageBox.critical(
                self, "Python Executable Error",
                f"Python executable '{sys.executable}' not found. This should not happen."
            )
            self.status_bar.showMessage("Critical: Python executable not found.", 0)
        except subprocess.TimeoutExpired:
             QMessageBox.warning(
                self, "Ultralytics Check Timeout",
                "Checking Ultralytics installation timed out. It might be slow or unresponsive."
            )
             self.status_bar.showMessage("Warning: Ultralytics check timed out.", 10000)
        except Exception as e:
            QMessageBox.warning(
                self, "Ultralytics Check Error",
                f"An error occurred while checking Ultralytics installation: {e}"
            )
            self.status_bar.showMessage(f"Error checking Ultralytics: {e}", 10000)


    def browse_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.yaml_path_edit.setText(file_path)

    def browse_custom_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Custom Model File", "", "PyTorch Model Files (*.pt)")
        if file_path:
            self.custom_model_path_edit.setText(file_path)

    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_dir_edit.setText(dir_path)

    def append_output(self, text):
        self.output_text.append(text.rstrip('\r\n')) # Avoids double newlines from some CLI tools
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def start_training(self):
        yaml_path = self.yaml_path_edit.text().strip()
        custom_model_path = self.custom_model_path_edit.text().strip()
        
        base_model = self.model_combo.currentText()
        chosen_model = custom_model_path if custom_model_path else base_model

        epochs = self.epochs_spinbox.value()
        imgsz = self.imgsz_spinbox.value()
        batch = self.batch_spinbox.value()
        
        project_dir = self.project_dir_edit.text().strip()
        experiment_name = self.experiment_name_edit.text().strip()

        device_full_name = self.device_combo.currentText()
        device = device_full_name.split(" ")[0] # Extract "cuda:0" or "cpu" from "cuda:0 (GeForce...)"

        optimizer = self.optimizer_combo.currentText()
        lr0 = self.lr0_spinbox.value()
        patience = self.patience_spinbox.value()
        weight_decay = self.weight_decay_spinbox.value()

        # --- Input Validation ---
        if not yaml_path or not Path(yaml_path).is_file():
            QMessageBox.warning(self, "Input Error", "Please select a valid dataset YAML file.")
            return
        if custom_model_path and not Path(custom_model_path).is_file():
            QMessageBox.warning(self, "Input Error", "Custom model path is specified but file not found.")
            return
        if imgsz % 32 != 0:
            QMessageBox.warning(self, "Input Error", "Image Size must be a multiple of 32.")
            return
        if not project_dir:
            QMessageBox.warning(self, "Input Error", "Please provide a Project Directory.")
            return
        if not experiment_name:
            QMessageBox.warning(self, "Input Error", "Please provide an Experiment Name.")
            return
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Training is already in progress.")
            return

        cmd = [
            sys.executable, "-m", "ultralytics.cli", "train",
            f"model={chosen_model}",
            f"data={yaml_path}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"project={project_dir}",
            f"name={experiment_name}",
            f"device={device}",
            f"optimizer={optimizer}",
            f"lr0={lr0}",
            f"patience={patience}",
            f"weight_decay={weight_decay}",
            # Add other parameters as needed:
            # "exist_ok=True", # To overwrite existing experiment with same name
            # "workers=8",
            # "verbose=True",
        ]

        self.output_text.clear()
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_controls_enabled(False)
        self.status_bar.showMessage("Training started...")

        self.training_thread = TrainingThread(cmd, total_epochs=epochs)
        self.training_thread.progress_signal.connect(self.append_output)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.progress_bar_signal.connect(self.update_progress_bar)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "Confirm Stop",
                                         "Are you sure you want to stop the training process?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.status_bar.showMessage("Stopping training...", 5000)
                self.stop_button.setEnabled(False)
                self.stop_button.setText("Stopping...")
                self.training_thread.stop()
        else:
            self.append_output("No active training process to stop.")
            self.status_bar.showMessage("No training process to stop.", 3000)

    def training_finished(self, exit_code, message):
        detailed_message = message
        if exit_code == 1:
            detailed_message = (
                f"{message}.\n\n"
                "This is a general error from the training process. Please carefully review the full output in the 'Training Output' console above for specific error messages from YOLOv8.\n\n"
                "Common things to check:\n"
                "1. Dataset: \n"
                "   - Is the path to your 'dataset.yaml' file correct in the GUI?\n"
                "   - Are all paths (train, val, names) inside your 'dataset.yaml' correct and accessible?\n"
                "   - Are your image files valid (not corrupted) and label files correctly formatted according to YOLO standards?\n"
                "   - Check for any 'FileNotFoundError' or dataset-related errors in the console output.\n"
                "2. Environment & Installation: \n"
                "   - Did the 'Ultralytics installation check' pass at startup? (See initial status bar message or pop-ups).\n"
                "   - Is Ultralytics (e.g., 'pip install ultralytics') and its dependencies (PyTorch, torchvision, pandas, numpy, etc.) correctly installed in the Python environment being used?\n"
                "   - Are versions of PyTorch, CUDA (if using GPU), and Ultralytics compatible? Outdated or mismatched versions can cause issues.\n"
                "   - Consider running in a clean virtual environment.\n"
                "3. Resources & Configuration:\n"
                "   - If using GPU: Is the correct GPU device selected? Are NVIDIA drivers and CUDA toolkit installed and compatible with PyTorch? (Check 'nvidia-smi' command output).\n"
                "   - Out of Memory (OOM): Look for 'OutOfMemoryError' or similar in the console. Try reducing 'Batch Size' or 'Image Size'.\n"
                "   - Invalid model file: If using a custom .pt model, ensure the file is a valid PyTorch model weights file and not corrupted.\n"
                "4. Training Parameters:\n"
                "   - Double-check all training parameters in the GUI for sensible values.\n\n"
                "The console output right before the training stops is often the key to diagnosing the specific problem."
            )
        
        QMessageBox.information(self, "Training Status", detailed_message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stop Training")
        self.set_controls_enabled(True)
        self.status_bar.showMessage(f"Training finished. {detailed_message}", 10000)

        if exit_code == 0: # Only offer to open if successful
            try:
                # Correctly form the path to the specific experiment
                results_dir = Path(self.project_dir_edit.text().strip()) / self.experiment_name_edit.text().strip()
                if results_dir.exists():
                    open_dir_reply = QMessageBox.question(self, "Training Complete",
                                                f"Training finished successfully.\nResults saved in:\n{results_dir}\n\nOpen this directory?",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                QMessageBox.StandardButton.Yes)
                    if open_dir_reply == QMessageBox.StandardButton.Yes:
                        if sys.platform == 'win32':
                            os.startfile(results_dir)
                        elif sys.platform == 'darwin':
                            subprocess.Popen(['open', str(results_dir)])
                        else:
                            subprocess.Popen(['xdg-open', str(results_dir)])
            except Exception as e:
                self.append_output(f"Could not open results directory: {e}")
                self.status_bar.showMessage(f"Error opening results directory: {e}", 5000)

        self.training_thread = None

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable input controls."""
        # Dataset & Model
        self.yaml_path_edit.setEnabled(enabled)
        self.yaml_browse_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.custom_model_path_edit.setEnabled(enabled)
        self.custom_model_browse_button.setEnabled(enabled)
        # Output Config
        self.project_dir_edit.setEnabled(enabled)
        self.project_dir_browse_button.setEnabled(enabled)
        self.experiment_name_edit.setEnabled(enabled)
        # Core Training Params
        self.epochs_spinbox.setEnabled(enabled)
        self.imgsz_spinbox.setEnabled(enabled)
        self.batch_spinbox.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        # Hyperparams
        self.optimizer_combo.setEnabled(enabled)
        self.lr0_spinbox.setEnabled(enabled)
        self.patience_spinbox.setEnabled(enabled)
        self.weight_decay_spinbox.setEnabled(enabled)


    def closeEvent(self, event):
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "Exit Confirmation",
                                         "Training is in progress. Are you sure you want to exit? This will stop the training.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.status_bar.showMessage("Exiting and stopping training...", 0)
                self.training_thread.stop()
                self.training_thread.wait(5000) # Wait up to 5s
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# --- Run the Application ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET) # Apply the dark theme
    
    # Optional: Fusion style can sometimes interact well or poorly with custom stylesheets
    # app.setStyle('Fusion')

    ex = YoloTrainerApp()
    ex.show()
    sys.exit(app.exec())