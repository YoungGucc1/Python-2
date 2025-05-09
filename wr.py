import sys
import os
import subprocess
import json
import yaml # pip install PyYAML
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QSpinBox, QComboBox, QFormLayout, QGroupBox, QMessageBox,
    QStatusBar
)
from PyQt6.QtCore import QProcess, Qt, QSettings

# --- Configuration ---
CONFIG_FILE = "trainer_config.json"
DEFAULT_YOLOV5_PATH = "yolov5" # Relative path to yolov5 cloned repo

class YoloTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training Wrapper")
        self.setGeometry(100, 100, 800, 700)

        self.process = None
        self.yolov5_train_script_path = "" # To be set if YOLOv5 is used

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Project Settings ---
        project_group = QGroupBox("Project")
        project_layout = QFormLayout()
        self.project_name_edit = QLineEdit("my_yolo_project")
        project_layout.addRow("Project Name:", self.project_name_edit)
        save_project_btn = QPushButton("Save Project Settings")
        save_project_btn.clicked.connect(self.save_project_settings)
        load_project_btn = QPushButton("Load Project Settings")
        load_project_btn.clicked.connect(self.load_project_settings)
        project_buttons_layout = QHBoxLayout()
        project_buttons_layout.addWidget(save_project_btn)
        project_buttons_layout.addWidget(load_project_btn)
        project_layout.addRow(project_buttons_layout)
        project_group.setLayout(project_layout)
        main_layout.addWidget(project_group)

        # --- Dataset Configuration ---
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout()
        self.dataset_yaml_edit = QLineEdit()
        self.dataset_yaml_edit.setPlaceholderText("Path to data.yaml")
        browse_dataset_btn = QPushButton("Browse...")
        browse_dataset_btn.clicked.connect(self.browse_dataset_yaml)
        dataset_path_layout = QHBoxLayout()
        dataset_path_layout.addWidget(self.dataset_yaml_edit)
        dataset_path_layout.addWidget(browse_dataset_btn)
        dataset_layout.addRow("Dataset YAML:", dataset_path_layout)
        self.dataset_info_label = QLabel("Dataset info will appear here.")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addRow(self.dataset_info_label)
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        # --- Model and Training Configuration ---
        model_train_group = QGroupBox("Model & Training Configuration")
        model_train_layout = QFormLayout()

        self.yolo_version_combo = QComboBox()
        self.yolo_version_combo.addItems(["YOLOv8", "YOLOv5"])
        # Define all model train widgets before connecting and calling update_yolov5_path_visibility
        
        self.yolov5_path_label = QLabel("YOLOv5 Repo Path:")
        self.yolov5_path_edit = QLineEdit(DEFAULT_YOLOV5_PATH)
        self.yolov5_path_browse_btn = QPushButton("Browse...")
        self.yolov5_path_browse_btn.clicked.connect(self.browse_yolov5_repo)
        yolov5_path_layout = QHBoxLayout()
        yolov5_path_layout.addWidget(self.yolov5_path_edit)
        yolov5_path_layout.addWidget(self.yolov5_path_browse_btn)
        self.yolov5_path_row_widget = QWidget() # To hide/show this row
        self.yolov5_path_row_widget.setLayout(yolov5_path_layout)
        
        self.base_model_edit = QLineEdit("yolov8n.pt") # Default for v8
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(-1, 256) # -1 for auto-batch
        self.batch_size_spin.setValue(16)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 2048)
        self.img_size_spin.setStep(32)
        self.img_size_spin.setValue(640)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cpu"]) # Add more if needed
        
        self.experiment_name_edit = QLineEdit("my_experiment")

        # Now add rows and connect signals
        model_train_layout.addRow("YOLO Version:", self.yolo_version_combo)
        model_train_layout.addRow(self.yolov5_path_label, self.yolov5_path_row_widget)
        model_train_layout.addRow("Base Model (.pt or .yaml):", self.base_model_edit)
        model_train_layout.addRow("Epochs:", self.epochs_spin)
        model_train_layout.addRow("Batch Size:", self.batch_size_spin)
        model_train_layout.addRow("Image Size (imgsz):", self.img_size_spin)
        model_train_layout.addRow("Device:", self.device_combo)
        model_train_layout.addRow("Experiment Name:", self.experiment_name_edit)
        
        self.yolo_version_combo.currentTextChanged.connect(self.update_yolov5_path_visibility)
        self.update_yolov5_path_visibility(self.yolo_version_combo.currentText()) # Initial state

        model_train_group.setLayout(model_train_layout)
        main_layout.addWidget(model_train_group)

        # --- Training Controls and Output ---
        train_control_group = QGroupBox("Training")
        train_control_layout = QVBoxLayout()

        self.start_train_btn = QPushButton("Start Training")
        self.start_train_btn.clicked.connect(self.start_training)
        self.stop_train_btn = QPushButton("Stop Training")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_train_btn)
        buttons_layout.addWidget(self.stop_train_btn)
        train_control_layout.addLayout(buttons_layout)

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setPlaceholderText("Training output will appear here...")
        train_control_layout.addWidget(self.output_console)

        self.results_path_label = QLabel("Results will be saved in: Not yet trained.")
        self.open_results_btn = QPushButton("Open Results Folder")
        self.open_results_btn.clicked.connect(self.open_results_folder)
        self.open_results_btn.setEnabled(False)
        results_layout = QHBoxLayout()
        results_layout.addWidget(self.results_path_label)
        results_layout.addWidget(self.open_results_btn)
        train_control_layout.addLayout(results_layout)

        train_control_group.setLayout(train_control_layout)
        main_layout.addWidget(train_control_group)

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready.")

        # Load last used settings if available
        self.load_app_settings()


    def update_yolov5_path_visibility(self, version_text):
        is_yolov5 = (version_text == "YOLOv5")
        self.yolov5_path_label.setVisible(is_yolov5)
        self.yolov5_path_row_widget.setVisible(is_yolov5)
        if is_yolov5:
            self.base_model_edit.setText("yolov5s.pt") # Default for v5
        else:
            self.base_model_edit.setText("yolov8n.pt") # Default for v8


    def browse_dataset_yaml(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML", "", "YAML Files (*.yaml *.yml)")
        if file_name:
            self.dataset_yaml_edit.setText(file_name)
            self.parse_dataset_yaml(file_name)

    def parse_dataset_yaml(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data_yaml_content = yaml.safe_load(f)

            nc = data_yaml_content.get('nc', 'N/A')
            names = data_yaml_content.get('names', [])
            train_path_str = data_yaml_content.get('train', 'N/A')
            val_path_str = data_yaml_content.get('val', 'N/A')
            test_path_str = data_yaml_content.get('test', None) # Optional

            yaml_file_abs_path = Path(file_path).resolve()
            yaml_dir = yaml_file_abs_path.parent

            # Determine the dataset root directory
            # Ultralytics uses a 'path' key in the YAML for the dataset root.
            # If 'path' is relative, it's relative to the YAML file's directory.
            # If no 'path' key, paths inside YAML (train, val) are relative to YAML's directory.
            dataset_root_str = data_yaml_content.get('path', None)
            if dataset_root_str:
                dataset_root_path = Path(dataset_root_str)
                if not dataset_root_path.is_absolute():
                    dataset_root = (yaml_dir / dataset_root_path).resolve()
                else:
                    dataset_root = dataset_root_path.resolve()
            else:
                dataset_root = yaml_dir

            # Resolve train, val, test paths.
            # These paths (train_path_str, etc.) are typically relative to dataset_root.
            # If a path string in the YAML is already absolute, Path objects handle it correctly.
            resolved_train_path, train_exists = "N/A", False
            if isinstance(train_path_str, str):
                path_obj = Path(train_path_str)
                resolved_train_path = (dataset_root / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
                train_exists = resolved_train_path.exists()

            resolved_val_path, val_exists = "N/A", False
            if isinstance(val_path_str, str):
                path_obj = Path(val_path_str)
                resolved_val_path = (dataset_root / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
                val_exists = resolved_val_path.exists()

            info_text = f"Dataset YAML: {yaml_file_abs_path}\n"
            info_text += f"Interpreted Dataset Root: {dataset_root}\n"
            info_text += f"Classes (nc): {nc}\n"
            info_text += f"Class Names: {', '.join(names)}\n"
            info_text += "--- Paths from YAML ---\n"
            info_text += f"Train (raw): '{train_path_str}' -> Resolved: {resolved_train_path} (Exists: {train_exists})\n"
            info_text += f"Val (raw): '{val_path_str}' -> Resolved: {resolved_val_path} (Exists: {val_exists})\n"

            resolved_test_path, test_exists = "N/A", False
            if test_path_str is not None and isinstance(test_path_str, str):
                path_obj = Path(test_path_str)
                resolved_test_path = (dataset_root / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
                test_exists = resolved_test_path.exists()
                info_text += f"Test (raw): '{test_path_str}' -> Resolved: {resolved_test_path} (Exists: {test_exists})\n"
            elif test_path_str is not None: # Present but not a string
                 info_text += f"Test (raw): '{test_path_str}' -> Invalid format, expected a path string.\n"


            if not train_exists or not val_exists:
                # Highlight if crucial paths are missing
                warning_style = "color: red; font-weight: bold;"
                self.dataset_info_label.setStyleSheet(warning_style)
                info_text += "\nWARNING: Critical dataset paths (train/val) are missing. Please check your data.yaml and file locations."
            else:
                self.dataset_info_label.setStyleSheet("") # Reset style

            self.dataset_info_label.setText(info_text)
            self.statusBar.showMessage(f"Dataset YAML '{yaml_file_abs_path.name}' parsed.", 5000)

        except Exception as e:
            error_message = f"Error parsing dataset YAML '{file_path}':\n{type(e).__name__}: {e}"
            self.dataset_info_label.setText(error_message)
            self.dataset_info_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Dataset YAML Error", error_message)

    def browse_yolov5_repo(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select YOLOv5 Repository Root")
        if dir_name:
            self.yolov5_path_edit.setText(dir_name)


    def validate_inputs(self):
        if not self.dataset_yaml_edit.text() or not os.path.exists(self.dataset_yaml_edit.text()):
            QMessageBox.warning(self, "Input Error", "Dataset YAML path is invalid.")
            return False
        if self.yolo_version_combo.currentText() == "YOLOv5":
            self.yolov5_train_script_path = Path(self.yolov5_path_edit.text()) / "train.py"
            if not self.yolov5_train_script_path.exists():
                QMessageBox.warning(self, "Input Error", f"YOLOv5 train.py not found at: {self.yolov5_train_script_path}")
                return False
        if not self.base_model_edit.text():
            QMessageBox.warning(self, "Input Error", "Base model cannot be empty.")
            return False
        if not self.experiment_name_edit.text():
            QMessageBox.warning(self, "Input Error", "Experiment name cannot be empty.")
            return False
        return True

    def start_training(self):
        if not self.validate_inputs():
            return

        self.output_console.clear()
        self.results_path_label.setText("Results will be saved in: Training...")
        self.open_results_btn.setEnabled(False)

        yolo_version = self.yolo_version_combo.currentText()
        # Resolve data_yaml path to absolute for consistency
        data_yaml_abs_path = str(Path(self.dataset_yaml_edit.text()).resolve())
        base_model = self.base_model_edit.text()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_size_spin.value()
        img_size = self.img_size_spin.value()
        device = self.device_combo.currentText()
        exp_name = self.experiment_name_edit.text()
        project_name = self.project_name_edit.text()

        cmd = []

        if yolo_version == "YOLOv8":
            try:
                subprocess.run(["yolo", "-h"], capture_output=True, check=True, text=True, shell=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                 QMessageBox.critical(self, "YOLOv8 Error",
                                     "Ultralytics 'yolo' command not found or not working. "
                                     "Please ensure it's installed and in your PATH.")
                 return

            cmd = [
                "yolo", "train",
                f"data={data_yaml_abs_path}", # Use absolute path
                f"model={base_model}",
                f"epochs={epochs}",
                f"batch={batch_size}",
                f"imgsz={img_size}",
                f"device={device}",
                f"project={project_name}",
                f"name={exp_name}"
            ]
        elif yolo_version == "YOLOv5":
            yolov5_root = Path(self.yolov5_path_edit.text())
            if not yolov5_root.is_dir():
                QMessageBox.warning(self, "YOLOv5 Error", f"YOLOv5 directory not found: {yolov5_root}")
                return

            cmd = [
                sys.executable,
                str(self.yolov5_train_script_path),
                "--data", data_yaml_abs_path, # Use absolute path
                "--weights", base_model,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--imgsz", str(img_size),
                "--device", device,
                "--project", project_name,
                "--name", exp_name
            ]
        else:
            QMessageBox.warning(self, "Error", "Invalid YOLO version selected.")
            return

        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels) # Easier for single console

        self.output_console.append(f"Starting training with command:\n{' '.join(cmd)}\n---")
        
        if yolo_version == "YOLOv5":
            self.process.setWorkingDirectory(str(Path(self.yolov5_path_edit.text()).resolve()))
            self.output_console.append(f"Working directory set to: {self.process.workingDirectory()}")
        
        try:
            self.process.start(cmd[0], cmd[1:]) # cmd[0] is program, cmd[1:] are arguments
            if not self.process.waitForStarted(5000): # Wait 5 seconds for process to start
                QMessageBox.critical(self, "Process Error", f"Failed to start process: {self.process.errorString()}")
                self.process = None
                return
        except Exception as e:
            QMessageBox.critical(self, "Process Error", f"Exception starting process: {e}")
            self.process = None
            return


        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.statusBar.showMessage("Training started...")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode(errors='ignore')
        self.output_console.moveCursor(self.output_console.textCursor().End)
        self.output_console.insertPlainText(data)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode(errors='ignore')
        self.output_console.moveCursor(self.output_console.textCursor().End)
        self.output_console.insertPlainText(data) # Merged, so could also come here

    def stop_training(self):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate() # or .kill() for forceful
            # self.process.waitForFinished(3000) # Wait a bit for it to terminate
            self.output_console.append("\n--- Training manually stopped by user. ---")
            self.statusBar.showMessage("Training stopped by user.", 5000)
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def process_finished(self, exitCode, exitStatus):
        self.output_console.append(f"\n--- Training finished. Exit Code: {exitCode}, Status: {exitStatus.name} ---")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        
        if exitCode == 0 and exitStatus == QProcess.ExitStatus.NormalExit:
            self.statusBar.showMessage("Training completed successfully!", 5000)
            # Construct results path (Ultralytics standard: runs/train/exp_name or project_name/exp_name)
            project_dir = self.project_name_edit.text()
            exp_name = self.experiment_name_edit.text()

            # YOLOv5 saves in its CWD, YOLOv8 in current CWD by default unless project is absolute
            if self.yolo_version_combo.currentText() == "YOLOv5":
                results_base_dir = Path(self.yolov5_path_edit.text()).resolve()
            else: # YOLOv8
                results_base_dir = Path.cwd() # Or handle if 'project' was absolute

            self.current_results_path = results_base_dir / project_dir / exp_name
            # Ultralytics sometimes adds a number if exp_name exists, e.g., exp_name2, exp_name3
            # A more robust way would be to parse the output for the exact path or find the latest dir.
            # For simplicity, we assume this direct path for now.
            # Check if the directory exists; if not, try to find the latest one.
            if not self.current_results_path.exists():
                potential_parent = results_base_dir / project_dir
                if potential_parent.exists():
                    subdirs = sorted([d for d in potential_parent.iterdir() if d.is_dir() and d.name.startswith(exp_name)], key=os.path.getmtime, reverse=True)
                    if subdirs:
                        self.current_results_path = subdirs[0]

            if self.current_results_path.exists():
                self.results_path_label.setText(f"Results: {self.current_results_path}")
                self.open_results_btn.setEnabled(True)
            else:
                self.results_path_label.setText(f"Results: Could not reliably determine path (expected near {self.current_results_path})")
                self.open_results_btn.setEnabled(False)
        else:
            self.statusBar.showMessage(f"Training failed or was interrupted. Exit Code: {exitCode}", 5000)
            self.results_path_label.setText("Results: Training did not complete successfully.")
            self.open_results_btn.setEnabled(False)
        self.process = None

    def open_results_folder(self):
        if hasattr(self, 'current_results_path') and self.current_results_path.exists():
            # Use platform-agnostic way to open folder
            if sys.platform == "win32":
                os.startfile(str(self.current_results_path))
            elif sys.platform == "darwin": # macOS
                subprocess.Popen(["open", str(self.current_results_path)])
            else: # linux and other UNIX
                subprocess.Popen(["xdg-open", str(self.current_results_path)])
        else:
            QMessageBox.information(self, "No Path", "Results path is not available or does not exist.")

    # --- App Settings and Project Settings Persistence ---
    def get_current_settings(self):
        """Gets all UI configurable settings into a dict"""
        return {
            "project_name": self.project_name_edit.text(),
            "dataset_yaml": self.dataset_yaml_edit.text(),
            "yolo_version": self.yolo_version_combo.currentText(),
            "yolov5_repo_path": self.yolov5_path_edit.text(),
            "base_model": self.base_model_edit.text(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "img_size": self.img_size_spin.value(),
            "device": self.device_combo.currentText(),
            "experiment_name": self.experiment_name_edit.text(),
        }

    def apply_settings(self, settings_dict):
        """Applies a dict of settings to the UI elements"""
        self.project_name_edit.setText(settings_dict.get("project_name", "my_yolo_project"))
        self.dataset_yaml_edit.setText(settings_dict.get("dataset_yaml", ""))
        if self.dataset_yaml_edit.text(): # Try to parse if path exists
            self.parse_dataset_yaml(self.dataset_yaml_edit.text())

        yolo_version = settings_dict.get("yolo_version", "YOLOv8")
        if yolo_version in [self.yolo_version_combo.itemText(i) for i in range(self.yolo_version_combo.count())]:
            self.yolo_version_combo.setCurrentText(yolo_version)
        self.update_yolov5_path_visibility(yolo_version) # Ensure correct visibility

        self.yolov5_path_edit.setText(settings_dict.get("yolov5_repo_path", DEFAULT_YOLOV5_PATH))
        self.base_model_edit.setText(settings_dict.get("base_model", "yolov8n.pt"))
        self.epochs_spin.setValue(settings_dict.get("epochs", 100))
        self.batch_size_spin.setValue(settings_dict.get("batch_size", 16))
        self.img_size_spin.setValue(settings_dict.get("img_size", 640))
        
        device = settings_dict.get("device", "cuda:0")
        if device in [self.device_combo.itemText(i) for i in range(self.device_combo.count())]:
            self.device_combo.setCurrentText(device)
        
        self.experiment_name_edit.setText(settings_dict.get("experiment_name", "my_experiment"))


    def save_project_settings(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project Configuration", "", "JSON Files (*.json)")
        if file_name:
            settings_to_save = self.get_current_settings()
            try:
                with open(file_name, 'w') as f:
                    json.dump(settings_to_save, f, indent=4)
                self.statusBar.showMessage(f"Project settings saved to {file_name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving", f"Could not save project settings: {e}")

    def load_project_settings(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project Configuration", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    loaded_settings = json.load(f)
                self.apply_settings(loaded_settings)
                self.statusBar.showMessage(f"Project settings loaded from {file_name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error Loading", f"Could not load project settings: {e}")

    def load_app_settings(self):
        """Load last used settings on startup from QSettings"""
        settings = QSettings("MyCompany", "YoloTrainerApp") # For storing app state, not project files
        last_config_json = settings.value("last_config", "{}", type=str)
        try:
            last_config = json.loads(last_config_json)
            if last_config: # Check if it's not an empty dict
                self.apply_settings(last_config)
                self.statusBar.showMessage("Loaded last used settings.", 2000)
        except json.JSONDecodeError:
            self.statusBar.showMessage("No valid last used settings found.", 2000)


    def closeEvent(self, event):
        """Save current UI settings on close"""
        current_settings = self.get_current_settings()
        settings = QSettings("MyCompany", "YoloTrainerApp")
        settings.setValue("last_config", json.dumps(current_settings))
        
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "Training is in progress. Are you sure you want to exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_training() # Attempt graceful shutdown
                self.process.waitForFinished(1000) # Give it a moment
                if self.process and self.process.state() == QProcess.ProcessState.Running:
                    self.process.kill() # Force kill if still running
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Optional: Set a style
    # app.setStyle("Fusion") 
    window = YoloTrainerApp()
    window.show()
    sys.exit(app.exec())