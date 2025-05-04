import sys
import os
import subprocess # Using subprocess for simplicity here, Python API is also an option
import threading
from pathlib import Path
import importlib.util # To check for torch without importing directly initially

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QComboBox, QTextEdit, QMessageBox,
    QGroupBox, QProgressBar, QDoubleSpinBox # Added ProgressBar and QDoubleSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon # Optional: for setting an icon

# --- Try to import torch ---
_torch_available = False
try:
    if importlib.util.find_spec("torch") is not None:
        import torch
        _torch_available = True
except ImportError:
    print("Torch not found. CUDA device detection will be unavailable.") # Or log this


# --- Configuration ---
APP_NAME = "YOLOv8 Trainer GUI"
# Find a suitable icon file (e.g., .ico or .png) if you have one
# APP_ICON_PATH = "path/to/your/icon.png"

# --- Worker Thread for Training ---
class TrainingThread(QThread):
    """
    Runs the YOLO training in a separate thread to avoid freezing the GUI.
    Emits signals for progress updates and completion.
    """
    progress_signal = pyqtSignal(str)       # Signal to send text output
    finished_signal = pyqtSignal(int, str)  # Signal with exit code and final message
    progress_bar_signal = pyqtSignal(int) # Signal for epoch progress (approximate)

    def __init__(self, cmd_args):
        super().__init__()
        self.cmd_args = cmd_args
        self.process = None
        self._is_running = True
        self.total_epochs = 1 # Default, will be updated

    def run(self):
        """Execute the training command."""
        self.progress_signal.emit("Starting training process...\n")
        self.progress_signal.emit(f"Command: {' '.join(self.cmd_args)}\n---\n")
        self._is_running = True
        exit_code = -1
        final_message = "An unexpected error occurred."
        epoch_current = 0

        try:
            # Extract total epochs for progress bar (assuming it's in cmd_args)
            try:
                epochs_arg_index = self.cmd_args.index('epochs') + 1
                self.total_epochs = int(self.cmd_args[epochs_arg_index])
            except (ValueError, IndexError):
                self.total_epochs = 1 # Cannot determine epochs, disable progress bar update

            # Using subprocess to capture output in real-time
            self.process = subprocess.Popen(
                self.cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                bufsize=1, # Line buffered
                universal_newlines=True,
                encoding='utf-8',
                errors='replace', # Handle potential encoding errors
                shell=False # Safer not to use shell=True
            )

            # Read output line by line
            while self._is_running:
                line = self.process.stdout.readline()
                if not line:
                    break # Process finished
                self.progress_signal.emit(line)

                # --- Approximate Progress Bar Update ---
                # This is basic, relies on YOLOv8's output format. Might need adjustment.
                if f'/{self.total_epochs}' in line and 'Epoch' in line:
                    try:
                        # Extract current epoch number (simple parsing)
                        parts = line.split()
                        for part in parts:
                            if f'/{self.total_epochs}' in part and part.count('/') == 1:
                                current_str = part.split('/')[0]
                                if current_str.isdigit():
                                    epoch_current = int(current_str)
                                    progress_percent = int((epoch_current / self.total_epochs) * 100)
                                    self.progress_bar_signal.emit(progress_percent)
                                    break # Found epoch info
                    except Exception:
                        pass # Ignore parsing errors

            self.process.wait() # Wait for the process to truly finish
            exit_code = self.process.returncode

            if not self._is_running and exit_code != 0 : # Check if stopped manually and not successful
                 final_message = "Training stopped by user."
                 self.progress_signal.emit("\n---\nTraining process stopped manually.\n")
            elif exit_code == 0:
                final_message = "Training completed successfully!"
                self.progress_bar_signal.emit(100) # Ensure progress bar reaches 100%
                self.progress_signal.emit("\n---\nTraining finished successfully.\n")
            else:
                final_message = f"Training failed with exit code: {exit_code}"
                self.progress_signal.emit(f"\n---\nTraining process failed (Exit Code: {exit_code}).\n")

        except FileNotFoundError:
             final_message = "Error: 'yolo' command not found. Is Ultralytics installed and in PATH?"
             self.progress_signal.emit(f"\nError: {final_message}\n")
             exit_code = -1
        except Exception as e:
            final_message = f"An error occurred: {e}"
            self.progress_signal.emit(f"\n---\nAn error occurred during training:\n{e}\n")
            exit_code = -1
        finally:
            self._is_running = False
            self.process = None # Clear process reference
            self.finished_signal.emit(exit_code, final_message)

    def stop(self):
        """Request the training process to stop."""
        if self.process and self._is_running:
            self.progress_signal.emit("\n---\nAttempting to stop training...\n")
            self._is_running = False
            try:
                # Attempt graceful termination first
                self.process.terminate()
                # Wait a bit, then force kill if still running
                try:
                    self.process.wait(timeout=5) # Wait 5 seconds
                except subprocess.TimeoutExpired:
                    self.progress_signal.emit("Process did not terminate gracefully, forcing kill.\n")
                    self.process.kill()
            except Exception as e:
                self.progress_signal.emit(f"Error while trying to stop process: {e}\n")
        else:
             self._is_running = False # Ensure flag is set even if process is not running


# --- Main Application Window ---
class YoloTrainerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.output_dir_base = "runs/train" # Default YOLO output
        self.available_devices = self._detect_devices()

        self.init_ui()

    def _detect_devices(self):
        """Detects available compute devices (CPU, CUDA)."""
        devices = ["cpu"]
        if _torch_available and torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                devices.append(f"cuda:{i}") # ({torch.cuda.get_device_name(i)})") # Optional: Add device name
        # print(f"Detected devices: {devices}") # Debugging
        return devices

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        # Set window icon if path is valid
        # if os.path.exists(APP_ICON_PATH):
        #     self.setWindowIcon(QIcon(APP_ICON_PATH))

        main_layout = QVBoxLayout(self)

        # --- Dataset Configuration ---
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QVBoxLayout()

        # YAML Path
        yaml_layout = QHBoxLayout()
        self.yaml_label = QLabel("Dataset YAML:")
        self.yaml_path_edit = QLineEdit()
        self.yaml_path_edit.setPlaceholderText("Click 'Browse' to select your dataset.yaml file")
        self.yaml_browse_button = QPushButton("Browse...")
        self.yaml_browse_button.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(self.yaml_label)
        yaml_layout.addWidget(self.yaml_path_edit)
        yaml_layout.addWidget(self.yaml_browse_button)
        dataset_layout.addLayout(yaml_layout)
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        # --- Training Parameters ---
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        # Model Selection (Replaced dropdown with direct file browse)
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Model (.pt File):") # Renamed label
        self.model_path_edit = QLineEdit() # Renamed line edit
        self.model_path_edit.setPlaceholderText("Browse to select the model .pt file...")
        self.model_browse_button = QPushButton("Browse...") # Renamed button
        self.model_browse_button.clicked.connect(self.browse_model) # Renamed browse function
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_browse_button)
        params_layout.addLayout(model_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        self.epochs_label = QLabel("Epochs:")
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(100)
        self.epochs_spinbox.setToolTip("Number of training epochs.")
        epochs_layout.addWidget(self.epochs_label)
        epochs_layout.addWidget(self.epochs_spinbox)
        params_layout.addLayout(epochs_layout)

        # Image Size
        imgsz_layout = QHBoxLayout()
        self.imgsz_label = QLabel("Image Size (px):")
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 4096) # Example range
        self.imgsz_spinbox.setSingleStep(32) # YOLO requires multiples of 32
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setToolTip("Target image size for training (must be multiple of 32).")
        imgsz_layout.addWidget(self.imgsz_label)
        imgsz_layout.addWidget(self.imgsz_spinbox)
        params_layout.addLayout(imgsz_layout)

        # Batch Size
        batch_layout = QHBoxLayout()
        self.batch_label = QLabel("Batch Size:")
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(-1, 512) # -1 for AutoBatch
        self.batch_spinbox.setValue(16)
        self.batch_spinbox.setToolTip("Number of images per batch (-1 for AutoBatch, requires more VRAM). Adjust based on GPU memory.")
        batch_layout.addWidget(self.batch_label)
        batch_layout.addWidget(self.batch_spinbox)
        params_layout.addLayout(batch_layout)

        # Project Name
        name_layout = QHBoxLayout()
        self.name_label = QLabel("Project Name:")
        self.name_edit = QLineEdit("yolo_training_run")
        self.name_edit.setToolTip("Name for the results folder (inside runs/train).")
        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_edit)
        params_layout.addLayout(name_layout)

        # Device
        device_layout = QHBoxLayout()
        self.device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        # self.device_combo.addItems(["cuda:0", "cpu"]) # Default to GPU 0 or CPU
        self.device_combo.addItems(self.available_devices) # Populate dynamically
        # You could add logic here to detect available GPUs if desired
        self.device_combo.setToolTip("Select compute device (CPU or available CUDA GPUs).")
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_combo)
        params_layout.addLayout(device_layout)

        # --- Advanced Parameters ---
        adv_params_layout = QHBoxLayout() # Use QHBoxLayout for side-by-side

        # Learning Rate (lr0)
        lr_layout = QVBoxLayout() # Group label and input vertically
        self.lr_label = QLabel("Learning Rate (lr0):")
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setRange(0.0, 1.0)
        self.lr_spinbox.setSingleStep(0.001)
        self.lr_spinbox.setValue(0.01) # Common default for AdamW
        self.lr_spinbox.setToolTip("Initial learning rate (e.g., 0.01 for AdamW, 0.1 for SGD).")
        lr_layout.addWidget(self.lr_label)
        lr_layout.addWidget(self.lr_spinbox)
        adv_params_layout.addLayout(lr_layout)

        # Optimizer
        opt_layout = QVBoxLayout()
        self.optimizer_label = QLabel("Optimizer:")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]) # Common optimizers + auto
        self.optimizer_combo.setCurrentText("auto")
        self.optimizer_combo.setToolTip("Training optimizer.")
        opt_layout.addWidget(self.optimizer_label)
        opt_layout.addWidget(self.optimizer_combo)
        adv_params_layout.addLayout(opt_layout)


        # Patience
        patience_layout = QVBoxLayout()
        self.patience_label = QLabel("Patience:")
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(0, 1000) # 0 means disabled
        self.patience_spinbox.setValue(50) # YOLOv8 default
        self.patience_spinbox.setToolTip("Epochs to wait for no improvement before early stopping (0 to disable).")
        patience_layout.addWidget(self.patience_label)
        patience_layout.addWidget(self.patience_spinbox)
        adv_params_layout.addLayout(patience_layout)

        params_layout.addLayout(adv_params_layout) # Add the row of advanced params


        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # --- Output Console ---
        output_group = QGroupBox("Training Output")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Training output will appear here...")
        self.output_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Keep long lines from wrapping
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group) # Add before progress bar

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Epoch Progress: %p%")
        main_layout.addWidget(self.progress_bar) # Add after output group


        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setStyleSheet("background-color: lightgreen;")

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False) # Disabled initially
        self.stop_button.setStyleSheet("background-color: lightcoral;")


        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.resize(700, 600) # Set a reasonable default size

    def browse_yaml(self):
        """Open a file dialog to select the dataset.yaml file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset YAML File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.yaml_path_edit.setText(file_path)

    def browse_model(self):
        """Open a file dialog to select the model .pt file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Model Files (*.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def append_output(self, text):
        """Append text to the output QTextEdit."""
        self.output_text.append(text)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum() # Auto-scroll
        )

    def update_progress_bar(self, value):
        """Update the progress bar value."""
        self.progress_bar.setValue(value)

    def start_training(self):
        """Validate inputs and start the training thread."""
        yaml_path = self.yaml_path_edit.text()
        model_path = self.model_path_edit.text().strip() # Get model path from the line edit
        epochs = self.epochs_spinbox.value()
        imgsz = self.imgsz_spinbox.value()
        batch = self.batch_spinbox.value()
        name = self.name_edit.text().strip()
        device = self.device_combo.currentText()
        lr0 = self.lr_spinbox.value()
        optimizer = self.optimizer_combo.currentText()
        patience = self.patience_spinbox.value()

        # --- Input Validation ---
        if not yaml_path or not Path(yaml_path).is_file():
            QMessageBox.warning(self, "Input Error", "Please select a valid dataset YAML file.")
            return
        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(self, "Input Error", "Please select a valid model (.pt) file.")
            return
        if imgsz % 32 != 0:
             QMessageBox.warning(self, "Input Error", "Image Size must be a multiple of 32.")
             return
        if not name:
            QMessageBox.warning(self, "Input Error", "Please provide a project name.")
            return
        if self.training_thread and self.training_thread.isRunning():
             QMessageBox.warning(self, "Busy", "Training is already in progress.")
             return

        # Use custom weights if provided, otherwise use selected standard model
        # model_source = custom_weights_path if custom_weights_path else model

        # --- Construct Command ---
        cmd = self._build_yolo_command(
            model=model_path, # Use the directly selected model path
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            device=device,
            lr0=lr0,
            optimizer=optimizer,
            patience=patience
        )
        if not cmd: # Handle potential error during command building
            return

        # Clear previous output and reset progress bar
        self.output_text.clear()
        self.progress_bar.setValue(0)

        # Disable start button, enable stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_controls_enabled(False) # Disable input fields during training

        # Create and start the thread
        self.training_thread = TrainingThread(cmd)
        self.training_thread.progress_signal.connect(self.append_output)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.progress_bar_signal.connect(self.update_progress_bar)
        self.training_thread.start()

    def _build_yolo_command(self, **kwargs):
        """Builds the list of arguments for the YOLO CLI command."""
        try:
            # Base command using sys.executable to be robust
            cmd_base = [sys.executable, "-m", "ultralytics.cli", "train"]

            # Required arguments (already validated in start_training)
            cmd_args = [
                f"model={kwargs['model']}",
                f"data={kwargs['data']}",
                f"epochs={kwargs['epochs']}",
                f"imgsz={kwargs['imgsz']}",
                f"batch={kwargs['batch']}",
                f"name={kwargs['name']}",
                f"device={kwargs['device']}",
            ]

            # Optional arguments - Add only if not default or 'auto'
            if kwargs.get('lr0') is not None: # Check if lr0 exists and add
                 cmd_args.append(f"lr0={kwargs['lr0']:.5f}") # Format learning rate

            optimizer = kwargs.get('optimizer', 'auto')
            if optimizer != 'auto':
                cmd_args.append(f"optimizer={optimizer}")

            patience = kwargs.get('patience', 50) # Default YOLO patience is 50
            # Only add patience if it's different from default AND not 0 (disabled)
            # Or always add if you prefer explicitness: cmd_args.append(f"patience={patience}")
            if patience != 50:
                 cmd_args.append(f"patience={patience}")


            # Add more optional parameters here as needed following the pattern:
            # if kwargs.get('some_param') is not None and kwargs['some_param'] != DEFAULT_VALUE:
            #    cmd_args.append(f"some_param={kwargs['some_param']}")

            return cmd_base + cmd_args
        except KeyError as e:
            QMessageBox.critical(self, "Command Build Error", f"Missing required parameter: {e}")
            return None
        except Exception as e:
             QMessageBox.critical(self, "Command Build Error", f"An unexpected error occurred building the command: {e}")
             return None

    def stop_training(self):
        """Signal the training thread to stop."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "Confirm Stop",
                                         "Are you sure you want to stop the training process?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.training_thread.stop()
                self.stop_button.setEnabled(False) # Prevent multiple stop clicks
                self.stop_button.setText("Stopping...")
        else:
            self.append_output("No active training process to stop.")


    def training_finished(self, exit_code, message):
        """Handle cleanup and UI updates when training finishes or is stopped."""
        QMessageBox.information(self, "Training Status", message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stop Training") # Reset button text
        self.set_controls_enabled(True) # Re-enable input fields

        # Optionally, open the results directory
        try:
            results_dir = Path(self.output_dir_base) / self.name_edit.text().strip()
            if results_dir.exists():
                 open_dir = QMessageBox.question(self, "Training Complete",
                                          f"Training finished.\nResults saved in:\n{results_dir}\n\nOpen this directory?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.Yes)
                 if open_dir == QMessageBox.StandardButton.Yes:
                     # Use platform-specific command to open directory
                     if sys.platform == 'win32':
                         os.startfile(results_dir)
                     elif sys.platform == 'darwin': # macOS
                         subprocess.Popen(['open', results_dir])
                     else: # Linux variants
                         subprocess.Popen(['xdg-open', results_dir])
        except Exception as e:
            self.append_output(f"Could not open results directory: {e}")

        self.training_thread = None # Clear thread reference


    def set_controls_enabled(self, enabled: bool):
        """Enable or disable input controls."""
        self.yaml_path_edit.setEnabled(enabled)
        self.yaml_browse_button.setEnabled(enabled)
        # self.model_combo.setEnabled(enabled) # Removed
        self.epochs_spinbox.setEnabled(enabled)
        self.imgsz_spinbox.setEnabled(enabled)
        self.batch_spinbox.setEnabled(enabled)
        self.name_edit.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        # Also toggle new controls
        self.model_path_edit.setEnabled(enabled) # Renamed
        self.model_browse_button.setEnabled(enabled) # Renamed
        # self.custom_weights_edit.setEnabled(enabled) # Removed
        # self.custom_weights_browse_button.setEnabled(enabled) # Removed
        self.lr_spinbox.setEnabled(enabled)
        self.optimizer_combo.setEnabled(enabled)
        self.patience_spinbox.setEnabled(enabled)

        # Special handling for model_combo based on custom_weights_edit (Removed)
        # if enabled and self.custom_weights_edit.text():
        #     self.model_combo.setEnabled(False)
        #     self.model_combo.setToolTip("Using custom weights file selected below.")
        # elif enabled:
        #      self.model_combo.setEnabled(True)
        #      self.model_combo.setToolTip("Select a standard pre-trained YOLO model.")


    def closeEvent(self, event):
        """Ensure thread is stopped cleanly when closing the window."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "Exit Confirmation",
                                         "Training is in progress. Are you sure you want to exit? This will stop the training.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.training_thread.stop()
                self.training_thread.wait(5000) # Wait up to 5s for thread to finish
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# --- Run the Application ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Optional: Apply a style for a potentially "comfier" look
    # app.setStyle('Fusion') # Or 'Windows', 'macOS' depending on platform
    ex = YoloTrainerApp()
    ex.show()
    sys.exit(app.exec())