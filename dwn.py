import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QFileDialog, QProgressBar, QMessageBox, QCheckBox, QGroupBox
)
from PyQt6.QtGui import QColor, QPalette, QFont, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# Try importing transformers, show error if not installed
try:
    from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
    from huggingface_hub import HfApi, HfFolder, login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Define color palette
class AppColors:
    PRIMARY = "#4A55A2"  # Deep blue
    SECONDARY = "#7895CB"  # Lighter blue
    BACKGROUND = "#F5F5F5"  # Light gray
    TEXT = "#333333"  # Dark gray
    ACCENT = "#A0BFE0"  # Light blue
    SUCCESS = "#4CAF50"  # Green
    ERROR = "#F44336"  # Red
    WARNING = "#FF9800"  # Orange

class ModelDownloadThread(QThread):
    progress_update = pyqtSignal(str)
    download_finished = pyqtSignal(bool, str)

    def __init__(self, model_name: str, token: str, model_type: str):
        super().__init__()
        self.model_name = model_name
        self.token = token
        self.model_type = model_type

    def run(self):
        if not TRANSFORMERS_AVAILABLE:
            self.download_finished.emit(False, "Required packages not installed")
            return

        try:
            if self.token:
                login(token=self.token)
                self.progress_update.emit("Logged in with provided token")
            else:
                token = HfFolder.get_token()
                if token:
                    self.progress_update.emit("Using existing cached token")
                else:
                    self.progress_update.emit("No token provided - proceeding with public models only")

            self.progress_update.emit(f"Downloading {self.model_name}...")
            
            # Download based on model type
            if self.model_type == "Model":
                model = AutoModel.from_pretrained(self.model_name)
                self.progress_update.emit("Model structure downloaded successfully")
            elif self.model_type == "Tokenizer":
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.progress_update.emit("Tokenizer downloaded successfully")
            elif self.model_type == "Feature Extractor":
                extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self.progress_update.emit("Feature extractor downloaded successfully")
            elif self.model_type == "Processor":
                processor = AutoProcessor.from_pretrained(self.model_name)
                self.progress_update.emit("Processor downloaded successfully")
            else:
                # Download all components
                self.progress_update.emit("Downloading model structure...")
                model = AutoModel.from_pretrained(self.model_name)
                
                self.progress_update.emit("Downloading tokenizer...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                except Exception as e:
                    self.progress_update.emit(f"Tokenizer not available: {str(e)}")
                
                self.progress_update.emit("Attempting to download feature extractor...")
                try:
                    extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                except Exception as e:
                    self.progress_update.emit(f"Feature extractor not available: {str(e)}")

            # Get cache location
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            self.progress_update.emit(f"Model cached at: {cache_dir}")
            
            self.download_finished.emit(True, "Download completed successfully")
            
        except Exception as e:
            error_msg = f"Error downloading model: {str(e)}"
            self.progress_update.emit(error_msg)
            self.download_finished.emit(False, error_msg)


class HuggingFaceDownloader(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Check if required packages are installed
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(
                self, 
                "Missing Dependencies",
                "Required packages not found. Please install with:\n\npip install transformers huggingface_hub torch"
            )
        
        self.setWindowTitle("Hugging Face Model Downloader")
        self.setMinimumSize(700, 500)
        self.setup_ui()
        self.download_thread = None
        
    def setup_ui(self):
        # Set global styles
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {AppColors.BACKGROUND}; }}
            QLabel {{ color: {AppColors.TEXT}; font-size: 14px; }}
            QLineEdit, QTextEdit, QComboBox {{ 
                background-color: white; 
                border: 1px solid {AppColors.SECONDARY}; 
                border-radius: 4px; 
                padding: 6px; 
                color: {AppColors.TEXT};
                font-size: 14px;
            }}
            QPushButton {{ 
                background-color: {AppColors.PRIMARY}; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                padding: 8px 16px; 
                font-size: 14px;
            }}
            QPushButton:hover {{ background-color: {AppColors.SECONDARY}; }}
            QPushButton:disabled {{ background-color: #CCCCCC; color: #666666; }}
            QGroupBox {{ 
                border: 1px solid {AppColors.SECONDARY}; 
                border-radius: 6px; 
                margin-top: 10px; 
                font-size: 14px;
                padding-top: 10px;
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px 0 5px; 
                color: {AppColors.PRIMARY};
                font-weight: bold;
            }}
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_label = QLabel("Hugging Face Model Downloader")
        header_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {AppColors.PRIMARY};")
        main_layout.addWidget(header_label)
        
        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)
        
        # Model Name Input
        model_name_layout = QHBoxLayout()
        model_name_label = QLabel("Model Name:")
        model_name_label.setFixedWidth(120)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("e.g., gpt2, bert-base-uncased, facebook/bart-large")
        model_name_layout.addWidget(model_name_label)
        model_name_layout.addWidget(self.model_name_input)
        model_layout.addLayout(model_name_layout)
        
        # Model Type Selection
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("Component:")
        model_type_label.setFixedWidth(120)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["All Components", "Model", "Tokenizer", "Feature Extractor", "Processor"])
        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)
        
        main_layout.addWidget(model_group)
        
        # Authentication Group
        auth_group = QGroupBox("Authentication")
        auth_layout = QVBoxLayout(auth_group)
        
        # Access Token Input
        token_layout = QHBoxLayout()
        token_label = QLabel("Access Token:")
        token_label.setFixedWidth(120)
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Optional: Enter your Hugging Face access token")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        token_layout.addWidget(token_label)
        token_layout.addWidget(self.token_input)
        auth_layout.addLayout(token_layout)
        
        # Remember Token Checkbox
        self.remember_token_checkbox = QCheckBox("Save token for future sessions")
        auth_layout.addWidget(self.remember_token_checkbox)
        
        main_layout.addWidget(auth_group)
        
        # Progress and Output Group
        output_group = QGroupBox("Download Progress")
        output_layout = QVBoxLayout(output_group)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        output_layout.addWidget(self.progress_bar)
        
        # Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(150)
        self.log_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: #FFFFFF;
                border: 1px solid {AppColors.SECONDARY};
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 13px;
            }}
        """)
        output_layout.addWidget(self.log_output)
        
        main_layout.addWidget(output_group)
        
        # Button Layout
        button_layout = QHBoxLayout()
        
        # View Cache Button
        self.view_cache_button = QPushButton("View Cache Directory")
        self.view_cache_button.clicked.connect(self.view_cache)
        button_layout.addWidget(self.view_cache_button)
        
        button_layout.addStretch()
        
        # Download Button
        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self.download_model)
        self.download_button.setMinimumWidth(150)
        self.download_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {AppColors.PRIMARY};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {AppColors.SECONDARY};
            }}
        """)
        button_layout.addWidget(self.download_button)
        
        main_layout.addLayout(button_layout)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Initial log message
        self.log("Ready to download Hugging Face models")
        
        # Load saved token if it exists
        if TRANSFORMERS_AVAILABLE:
            token = HfFolder.get_token()
            if token:
                self.token_input.setText(token)
                self.remember_token_checkbox.setChecked(True)
                self.log("Loaded saved token from cache")
    
    def log(self, message: str):
        """Add a message to the log output."""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def download_model(self):
        """Start the model download process."""
        model_name = self.model_name_input.text().strip()
        token = self.token_input.text().strip()
        model_type = self.model_type_combo.currentText()
        
        if not model_name:
            QMessageBox.warning(self, "Input Error", "Please enter a model name")
            return
        
        # Save token if requested
        if TRANSFORMERS_AVAILABLE and self.remember_token_checkbox.isChecked() and token:
            HfFolder.save_token(token)
            self.log("Token saved for future sessions")
        
        # Clear previous logs
        self.log_output.clear()
        self.log(f"Starting download for: {model_name}")
        
        # Update UI
        self.download_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # Start download thread
        self.download_thread = ModelDownloadThread(model_name, token, model_type)
        self.download_thread.progress_update.connect(self.log)
        self.download_thread.download_finished.connect(self.on_download_finished)
        self.download_thread.start()
    
    def on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        self.progress_bar.setVisible(False)
        self.download_button.setEnabled(True)
        
        if success:
            self.log(f"✅ {message}")
        else:
            self.log(f"❌ {message}")
            QMessageBox.warning(self, "Download Error", message)
    
    def view_cache(self):
        """Open the Hugging Face cache directory."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            self.log(f"Created cache directory: {cache_dir}")
        
        if sys.platform == 'win32':
            os.startfile(cache_dir)
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['open', cache_dir])
        else:  # Linux
            import subprocess
            subprocess.run(['xdg-open', cache_dir])
        
        self.log(f"Opened cache directory: {cache_dir}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HuggingFaceDownloader()
    window.show()
    sys.exit(app.exec())