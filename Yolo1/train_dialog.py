import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QComboBox,
    QDialogButtonBox, QPushButton, QFileDialog, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings

class TrainDialog(QDialog):
    def __init__(self, parent=None, current_dataset_yaml_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Train YOLO Model")
        self.setMinimumWidth(450)

        self.settings = QSettings()

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- Base Model Selection ---
        self.base_model_combo = QComboBox()
        self.base_model_options = {
            "YOLOv8 Nano (yolov8n.pt)": "yolov8n.pt",
            "YOLOv8 Small (yolov8s.pt)": "yolov8s.pt",
            "YOLOv8 Medium (yolov8m.pt)": "yolov8m.pt",
            "YOLOv8 Large (yolov8l.pt)": "yolov8l.pt",
            "YOLOv8 ExtraLarge (yolov8x.pt)": "yolov8x.pt",
            "Custom Path...": "custom"
        }
        self.base_model_combo.addItems(self.base_model_options.keys())
        self.base_model_combo.currentTextChanged.connect(self._on_base_model_changed)
        form_layout.addRow("Base Model:", self.base_model_combo)

        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("Path to custom .pt model")
        self.custom_model_path_btn = QPushButton("Browse...")
        self.custom_model_path_btn.clicked.connect(self._browse_custom_model)
        form_layout.addRow("Custom Model Path:", self.custom_model_path_edit) # Will be hidden initially
        form_layout.addRow("", self.custom_model_path_btn) # Will be hidden initially

        # --- Dataset YAML Path ---
        self.dataset_yaml_label = QLineEdit(current_dataset_yaml_path if current_dataset_yaml_path else "N/A - Save dataset first")
        self.dataset_yaml_label.setReadOnly(True) # Display only
        self.dataset_yaml_label.setToolTip("This is determined by the 'Save Dataset' action. Ensure your dataset is saved and a data.yaml is present.")
        form_layout.addRow("Dataset YAML:", self.dataset_yaml_label)

        # --- Training Parameters ---
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000); self.epochs_spin.setValue(self.settings.value("train/epochs", 100, type=int))
        form_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128); self.batch_size_spin.setValue(self.settings.value("train/batch_size", 16, type=int))
        self.batch_size_spin.setToolTip("Set to -1 for auto-batch size (experimental, requires ultralytics features).")
        # For auto-batch, ultralytics uses batch=-1. We can map a UI value (e.g. 0 or -1) if desired. For now, 1-128.
        form_layout.addRow("Batch Size:", self.batch_size_spin)

        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(32, 2048); self.image_size_spin.setStepType(QSpinBox.StepType.AdaptiveDecimalStepType)
        self.image_size_spin.setValue(self.settings.value("train/image_size", 640, type=int))
        self.image_size_spin.setToolTip("Target image size for training (e.g., 640). Must be multiple of 32.")
        form_layout.addRow("Image Size (pixels):", self.image_size_spin)
        
        self.device_combo = QComboBox()
        # Ideally, check for CUDA availability here if possible, or let ultralytics handle it.
        # For now, simple CPU/GPU. Ultralytics uses 'cpu' or '0', '1' etc for CUDA devices.
        self.device_combo.addItems(["cpu", "cuda:0 (if available)"]) # User needs to know if cuda:0 is valid
        self.device_combo.setCurrentText(self.settings.value("train/device", "cpu", type=str))
        form_layout.addRow("Device:", self.device_combo)
        
        self.project_name_edit = QLineEdit(self.settings.value("train/project_name", "runs/train", type=str))
        self.project_name_edit.setPlaceholderText("e.g., runs/train")
        form_layout.addRow("Project Name (Output Dir):", self.project_name_edit)

        self.run_name_edit = QLineEdit(self.settings.value("train/run_name", "exp", type=str))
        self.run_name_edit.setPlaceholderText("e.g., exp, my_custom_run")
        form_layout.addRow("Run Name (Sub-Dir):", self.run_name_edit)


        layout.addLayout(form_layout)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Start Training")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._on_base_model_changed(self.base_model_combo.currentText()) # Initial hide/show custom path

        # Restore last used base model choice
        last_base_model_text = self.settings.value("train/baseModelText", "YOLOv8 Nano (yolov8n.pt)", type=str)
        if last_base_model_text in self.base_model_options:
            self.base_model_combo.setCurrentText(last_base_model_text)
            if self.base_model_options[last_base_model_text] == "custom":
                self.custom_model_path_edit.setText(self.settings.value("train/customModelPath", "", type=str))


    def _on_base_model_changed(self, text: str):
        is_custom = (self.base_model_options.get(text) == "custom")
        self.custom_model_path_edit.setVisible(is_custom)
        self.custom_model_path_btn.setVisible(is_custom)
        # Adjust label for QFormLayout for proper hiding
        label_widget_c_path = self.layout().itemAt(0).layout().labelForField(self.custom_model_path_edit)
        if label_widget_c_path: label_widget_c_path.setVisible(is_custom)
        label_widget_c_btn = self.layout().itemAt(0).layout().labelForField(self.custom_model_path_btn)
        if label_widget_c_btn: label_widget_c_btn.setVisible(is_custom)


    def _browse_custom_model(self):
        last_dir = self.settings.value("train/lastCustomModelDir", os.path.expanduser("~"), type=str)
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Custom YOLO Model", last_dir, "PyTorch Models (*.pt)")
        if file_path:
            self.custom_model_path_edit.setText(file_path)
            self.settings.setValue("train/lastCustomModelDir", os.path.dirname(file_path))

    def _validate_parameters(self) -> bool:
        # Dataset YAML
        yaml_path = self.dataset_yaml_label.text()
        if not yaml_path or yaml_path == "N/A - Save dataset first" or not os.path.isfile(yaml_path):
            QMessageBox.warning(self, "Validation Error", "Dataset YAML path is invalid or not set. Please save your dataset first.")
            return False
        
        # Base Model
        selected_model_key = self.base_model_combo.currentText()
        base_model_val = self.base_model_options.get(selected_model_key)
        if base_model_val == "custom":
            custom_path = self.custom_model_path_edit.text()
            if not custom_path or not custom_path.endswith(".pt") or not os.path.isfile(custom_path):
                QMessageBox.warning(self, "Validation Error", "Custom model path is invalid or does not point to a .pt file.")
                return False
        elif not base_model_val: # Should not happen with ComboBox
            QMessageBox.warning(self, "Validation Error", "No valid base model selected.")
            return False

        # Image Size
        img_size = self.image_size_spin.value()
        if img_size % 32 != 0:
            QMessageBox.warning(self, "Validation Error", "Image size must be a multiple of 32.")
            return False
            
        # Project and Run names should not be empty
        if not self.project_name_edit.text().strip():
            QMessageBox.warning(self, "Validation Error", "Project Name cannot be empty.")
            return False
        if not self.run_name_edit.text().strip():
            QMessageBox.warning(self, "Validation Error", "Run Name cannot be empty.")
            return False

        return True

    def accept(self):
        if self._validate_parameters():
            # Save settings before accepting
            self.settings.setValue("train/baseModelText", self.base_model_combo.currentText())
            if self.base_model_options.get(self.base_model_combo.currentText()) == "custom":
                self.settings.setValue("train/customModelPath", self.custom_model_path_edit.text())
            self.settings.setValue("train/epochs", self.epochs_spin.value())
            self.settings.setValue("train/batch_size", self.batch_size_spin.value())
            self.settings.setValue("train/image_size", self.image_size_spin.value())
            self.settings.setValue("train/device", self.device_combo.currentText().split(" ")[0]) # Get 'cpu' or 'cuda:0'
            self.settings.setValue("train/project_name", self.project_name_edit.text().strip())
            self.settings.setValue("train/run_name", self.run_name_edit.text().strip())
            super().accept()
        # else: user has been warned by _validate_parameters

    def get_parameters(self) -> dict | None:
        if self.result() == QDialog.DialogCode.Accepted:
            base_model_val = self.base_model_options.get(self.base_model_combo.currentText())
            if base_model_val == "custom":
                model_to_train = self.custom_model_path_edit.text()
            else:
                model_to_train = base_model_val
            
            return {
                "model": model_to_train, # e.g., "yolov8n.pt" or "/path/to/custom.pt"
                "data": self.dataset_yaml_label.text(), # Path to data.yaml
                "epochs": self.epochs_spin.value(),
                "batch": self.batch_size_spin.value(),
                "imgsz": self.image_size_spin.value(),
                "device": self.device_combo.currentText().split(" ")[0], # 'cpu' or 'cuda:0'
                "project": self.project_name_edit.text().strip(), # Output directory for runs
                "name": self.run_name_edit.text().strip(), # Specific run name
                # Add other ultralytics params as needed: patience, optimizer, lr0, etc.
            }
        return None

if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # Example of how AppLogic might pass the yaml path
    example_yaml_path = os.path.abspath("./data/test_dataset/dataset.yaml") # dummy path
    dialog = TrainDialog(current_dataset_yaml_path=example_yaml_path)
    if dialog.exec():
        params = dialog.get_parameters()
        print("Training Parameters:", params)
    else:
        print("Training cancelled.")
    # sys.exit(app.exec()) # Not needed if just testing dialog 