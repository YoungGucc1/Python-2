import sys
import os
import json
import random
import requests
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QComboBox, QTextEdit, QFileDialog, QProgressBar, QCheckBox,
    QGroupBox, QTabWidget, QScrollArea, QSplitter, QListWidget,
    QMessageBox, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QIcon

# Import required diffusers components
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

# Theme colors
DARK_BLUE = "#1e293b"
MEDIUM_BLUE = "#334155"
LIGHT_BLUE = "#64748b"
ACCENT_BLUE = "#3b82f6"
ACCENT_PURPLE = "#8b5cf6"
TEXT_COLOR = "#f8fafc"
BG_COLOR = "#0f172a"


class DownloadModelThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, model_id, variant=None, local_dir=None):
        super().__init__()
        self.model_id = model_id
        self.variant = variant
        self.local_dir = local_dir or "models"
        
    def run(self):
        try:
            self.progress_signal.emit(f"Downloading model {self.model_id}...")
            
            if not os.path.exists(self.local_dir):
                os.makedirs(self.local_dir)
                
            if self.variant:
                # Download specific file variant
                model_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=self.variant,
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                self.progress_signal.emit(f"Downloaded {self.variant} successfully!")
            else:
                # Download the entire repository
                model_path = snapshot_download(
                    repo_id=self.model_id,
                    local_dir=os.path.join(self.local_dir, self.model_id.split('/')[-1]),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                self.progress_signal.emit(f"Downloaded entire model repository successfully!")
                
            self.finished_signal.emit(True, model_path)
            
        except Exception as e:
            self.progress_signal.emit(f"Error downloading model: {str(e)}")
            self.finished_signal.emit(False, str(e))


class GenerateImagesThread(QThread):
    progress_signal = pyqtSignal(str)
    image_signal = pyqtSignal(QImage, str)
    finished_signal = pyqtSignal()
    
    def __init__(self, model_path, config):
        super().__init__()
        self.model_path = model_path
        self.config = config
        
    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu"
            self.progress_signal.emit(f"Loading model on {device}...")
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.to(device)
            
            output_dir = self.config["output_dir"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            total_prompts = len(self.config["prompts"])
            
            for idx, prompt in enumerate(self.config["prompts"]):
                if not prompt.strip():
                    continue
                    
                self.progress_signal.emit(f"Generating image {idx+1}/{total_prompts}: {prompt[:50]}...")
                
                # Set the seed if specified, otherwise random
                if self.config["seed"] != -1:
                    generator = torch.Generator(device=device).manual_seed(self.config["seed"])
                    seed = self.config["seed"]
                else:
                    seed = random.randint(0, 2147483647)
                    generator = torch.Generator(device=device).manual_seed(seed)
                
                # Generate the image
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=self.config["negative_prompt"],
                    height=self.config["height"],
                    width=self.config["width"],
                    num_inference_steps=self.config["steps"],
                    guidance_scale=self.config["guidance_scale"],
                    generator=generator,
                ).images[0]
                
                # Prepare filename with timestamp and seed
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}_seed{seed}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save the image
                image.save(filepath)
                
                # Convert PIL image to QImage for display
                img_array = image.convert("RGBA")
                data = img_array.tobytes("raw", "RGBA")
                qimage = QImage(data, img_array.width, img_array.height, QImage.Format.Format_RGBA8888)
                
                metadata = {
                    "prompt": prompt,
                    "seed": seed,
                    "steps": self.config["steps"],
                    "guidance_scale": self.config["guidance_scale"],
                    "height": self.config["height"],
                    "width": self.config["width"],
                    "negative_prompt": self.config["negative_prompt"],
                }
                
                # Save metadata alongside the image
                with open(filepath.replace(".png", ".json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Emit the image for display
                self.image_signal.emit(qimage, filepath)
            
            self.progress_signal.emit("Generation complete!")
            self.finished_signal.emit()
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating images: {str(e)}")
            self.finished_signal.emit()


class FluxImageGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLUX.1 Image Generator")
        self.setMinimumSize(1000, 800)
        
        # Initialize variables
        self.model_path = None
        self.download_thread = None
        self.generate_thread = None
        self.images_generated = []
        
        # Check for GPU
        self.has_cuda = torch.cuda.is_available()
        
        # Set application style
        self.setup_style()
        
        # Create the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs content
        self.setup_model_tab()
        self.setup_generation_tab()
        self.setup_gallery_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Check for required libraries
        if not HAS_DIFFUSERS:
            self.show_error_message(
                "Missing Dependencies",
                "Required packages are not installed.\n\nPlease install with:\npip install diffusers transformers accelerate huggingface_hub"
            )
        
        # Display GPU status
        if not self.has_cuda:
            self.status_bar.showMessage("GPU not detected. Using CPU (generation will be slow).")
        else:
            self.status_bar.showMessage(f"GPU detected: {torch.cuda.get_device_name()}")
            
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLUX.1 Image Generator")
        self.setMinimumSize(1000, 800)
        
        # Initialize variables
        self.model_path = None
        self.download_thread = None
        self.generate_thread = None
        self.images_generated = []
        
        # Check for GPU - MUST be initialized before setup_generation_tab is called
        self.has_cuda = torch.cuda.is_available()
        
        # Set application style
        self.setup_style()
        
        # Create the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs content - make sure has_cuda is defined before this
        self.setup_model_tab()
        self.setup_generation_tab()  # This uses self.has_cuda
        self.setup_gallery_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Check for required libraries
        if not HAS_DIFFUSERS:
            self.show_error_message(
                "Missing Dependencies",
                "Required packages are not installed.\n\nPlease install with:\npip install diffusers transformers accelerate huggingface_hub"
            )
        
        # Display GPU status
        if not self.has_cuda:
            self.status_bar.showMessage("GPU not detected. Using CPU (generation will be slow).")
        else:
            self.status_bar.showMessage(f"GPU detected: {torch.cuda.get_device_name()}")
                
    def setup_style(self):
        # Set the application style
        app = QApplication.instance()
        
        # Set the palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_COLOR))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Base, QColor(DARK_BLUE))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(MEDIUM_BLUE))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(DARK_BLUE))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Button, QColor(MEDIUM_BLUE))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT_BLUE))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_PURPLE))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_COLOR))
        
        app.setPalette(palette)
        
        # Set the stylesheet
        stylesheet = f"""
        QMainWindow, QWidget {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
        QTabWidget::pane {{ background-color: {DARK_BLUE}; border: 1px solid {LIGHT_BLUE}; border-radius: 4px; }}
        QTabBar::tab {{ background-color: {MEDIUM_BLUE}; color: {TEXT_COLOR}; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
        QTabBar::tab:selected {{ background-color: {DARK_BLUE}; border-bottom-color: {DARK_BLUE}; }}
        QPushButton {{ background-color: {ACCENT_BLUE}; color: {TEXT_COLOR}; padding: 6px 12px; border: none; border-radius: 4px; }}
        QPushButton:hover {{ background-color: {ACCENT_PURPLE}; }}
        QPushButton:disabled {{ background-color: {LIGHT_BLUE}; color: {MEDIUM_BLUE}; }}
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color: {MEDIUM_BLUE}; color: {TEXT_COLOR}; padding: 6px; border: 1px solid {LIGHT_BLUE}; border-radius: 4px; }}
        QGroupBox {{ background-color: {DARK_BLUE}; border: 1px solid {LIGHT_BLUE}; border-radius: 4px; margin-top: 16px; padding-top: 16px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        QProgressBar {{ background-color: {DARK_BLUE}; border: 1px solid {LIGHT_BLUE}; border-radius: 4px; text-align: center; }}
        QProgressBar::chunk {{ background-color: {ACCENT_BLUE}; }}
        """
        
        app.setStyleSheet(stylesheet)
    
    def setup_model_tab(self):
        # Create the model tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        # Model download group
        download_group = QGroupBox("Model Download")
        download_layout = QVBoxLayout(download_group)
        
        # Model ID input
        model_id_layout = QHBoxLayout()
        model_id_label = QLabel("HuggingFace Model ID:")
        model_id_input = QLineEdit("city96/FLUX.1-dev-gguf")
        model_id_layout.addWidget(model_id_label)
        model_id_layout.addWidget(model_id_input)
        download_layout.addLayout(model_id_layout)
        
        # Model variant
        variant_layout = QHBoxLayout()
        variant_label = QLabel("Model Variant (optional):")
        variant_input = QLineEdit()
        variant_input.setPlaceholderText("Leave empty to download entire repository")
        variant_layout.addWidget(variant_label)
        variant_layout.addWidget(variant_input)
        download_layout.addLayout(variant_layout)
        
        # Download directory
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Download Directory:")
        self.dir_input = QLineEdit("models")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_download_dir)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_button)
        download_layout.addLayout(dir_layout)
        
        # Download button
        download_button = QPushButton("Download Model")
        download_button.clicked.connect(lambda: self.download_model(model_id_input.text(), variant_input.text()))
        download_layout.addWidget(download_button)
        
        # Status label
        self.download_status = QLabel("Ready to download")
        download_layout.addWidget(self.download_status)
        
        model_layout.addWidget(download_group)
        
        # Local model selection group
        local_group = QGroupBox("Local Model Selection")
        local_layout = QVBoxLayout(local_group)
        
        # Path input
        path_layout = QHBoxLayout()
        path_label = QLabel("Model Path:")
        self.path_input = QLineEdit()
        browse_model_button = QPushButton("Browse")
        browse_model_button.clicked.connect(self.browse_model_path)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_model_button)
        local_layout.addLayout(path_layout)
        
        # Load button
        load_button = QPushButton("Load Selected Model")
        load_button.clicked.connect(self.load_model)
        local_layout.addWidget(load_button)
        
        # Status label
        self.load_status = QLabel("No model loaded")
        local_layout.addWidget(self.load_status)
        
        model_layout.addWidget(local_group)
        model_layout.addStretch()
        
        self.tabs.addTab(model_tab, "1. Select Model")
        
    def setup_generation_tab(self):
        # Create the generation tab
        generation_tab = QWidget()
        generation_layout = QVBoxLayout(generation_tab)
        
        # Create a scroll area for the generation tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Prompt inputs
        prompt_group = QGroupBox("Prompts (One per line)")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompts here, one per line for batch processing...")
        prompt_layout.addWidget(self.prompt_input)
        
        # Negative prompt
        neg_prompt_layout = QHBoxLayout()
        neg_prompt_label = QLabel("Negative Prompt:")
        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setPlaceholderText("Elements to avoid in the generated images...")
        neg_prompt_layout.addWidget(neg_prompt_label)
        neg_prompt_layout.addWidget(self.neg_prompt_input)
        prompt_layout.addLayout(neg_prompt_layout)
        
        scroll_layout.addWidget(prompt_group)
        
        # Generation parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout(params_group)
        
        # Step count
        steps_label = QLabel("Steps:")
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 150)
        self.steps_input.setValue(30)
        params_layout.addWidget(steps_label, 0, 0)
        params_layout.addWidget(self.steps_input, 0, 1)
        
        # Guidance scale
        guidance_label = QLabel("Guidance Scale:")
        self.guidance_input = QDoubleSpinBox()
        self.guidance_input.setRange(1.0, 20.0)
        self.guidance_input.setValue(7.5)
        self.guidance_input.setSingleStep(0.5)
        params_layout.addWidget(guidance_label, 0, 2)
        params_layout.addWidget(self.guidance_input, 0, 3)
        
        # Width
        width_label = QLabel("Width:")
        self.width_input = QSpinBox()
        self.width_input.setRange(256, 1024)
        self.width_input.setValue(512)
        self.width_input.setSingleStep(8)
        params_layout.addWidget(width_label, 1, 0)
        params_layout.addWidget(self.width_input, 1, 1)
        
        # Height
        height_label = QLabel("Height:")
        self.height_input = QSpinBox()
        self.height_input.setRange(256, 1024)
        self.height_input.setValue(512)
        self.height_input.setSingleStep(8)
        params_layout.addWidget(height_label, 1, 2)
        params_layout.addWidget(self.height_input, 1, 3)
        
        # Seed
        seed_label = QLabel("Seed (-1 for random):")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(-1, 2147483647)
        self.seed_input.setValue(-1)
        params_layout.addWidget(seed_label, 2, 0)
        params_layout.addWidget(self.seed_input, 2, 1)
        
        # Device selection
        device_label = QLabel("Use CUDA:")
        self.device_checkbox = QCheckBox()
        self.device_checkbox.setChecked(self.has_cuda)
        self.device_checkbox.setEnabled(self.has_cuda)
        params_layout.addWidget(device_label, 2, 2)
        params_layout.addWidget(self.device_checkbox, 2, 3)
        
        scroll_layout.addWidget(params_group)
        
        # Output directory
        output_group = QGroupBox("Output Settings")
        output_layout = QHBoxLayout(output_group)
        output_label = QLabel("Output Directory:")
        self.output_input = QLineEdit("output")
        browse_output_button = QPushButton("Browse")
        browse_output_button.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(browse_output_button)
        
        scroll_layout.addWidget(output_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Images")
        self.generate_button.setEnabled(False)  # Disabled until model is loaded
        self.generate_button.clicked.connect(self.generate_images)
        scroll_layout.addWidget(self.generate_button)
        
        # Status label
        self.generation_status = QLabel("No model loaded")
        scroll_layout.addWidget(self.generation_status)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        generation_layout.addWidget(scroll_area)
        
        self.tabs.addTab(generation_tab, "2. Generate Images")
    
    def setup_gallery_tab(self):
        # Create the gallery tab
        gallery_tab = QWidget()
        gallery_layout = QVBoxLayout(gallery_tab)
        
        # Create a splitter for the gallery
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(200)
        self.image_list.itemClicked.connect(self.show_selected_image)
        splitter.addWidget(self.image_list)
        
        # Image display area
        image_display_widget = QWidget()
        image_display_layout = QVBoxLayout(image_display_widget)
        
        # Image preview
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_display_layout.addWidget(self.image_preview)
        
        # Image metadata
        self.image_metadata = QTextEdit()
        self.image_metadata.setReadOnly(True)
        self.image_metadata.setMaximumHeight(150)
        image_display_layout.addWidget(self.image_metadata)
        
        splitter.addWidget(image_display_widget)
        splitter.setSizes([200, 600])
        
        gallery_layout.addWidget(splitter)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Gallery")
        refresh_button.clicked.connect(self.refresh_gallery)
        gallery_layout.addWidget(refresh_button)
        
        self.tabs.addTab(gallery_tab, "3. Gallery")
    
    def browse_download_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Download Directory")
        if directory:
            self.dir_input.setText(directory)
    
    def browse_model_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if directory:
            self.path_input.setText(directory)
    
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_input.setText(directory)
    
    def download_model(self, model_id, variant):
        if not model_id:
            self.show_error_message("Error", "Please enter a valid model ID")
            return
        
        if not HAS_DIFFUSERS:
            self.show_error_message(
                "Missing Dependencies",
                "Required packages are not installed.\n\nPlease install with:\npip install diffusers transformers accelerate huggingface_hub"
            )
            return
        
        # Disable button during download
        sender = self.sender()
        sender.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Start download thread
        self.download_thread = DownloadModelThread(
            model_id=model_id,
            variant=variant if variant else None,
            local_dir=self.dir_input.text()
        )
        self.download_thread.progress_signal.connect(self.update_download_status)
        self.download_thread.finished_signal.connect(lambda success, path: self.download_completed(success, path, sender))
        self.download_thread.start()
    
    def update_download_status(self, message):
        self.download_status.setText(message)
        self.status_bar.showMessage(message)
    
    def download_completed(self, success, path, button):
        # Re-enable button
        button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        if success:
            self.download_status.setText(f"Model downloaded successfully to: {path}")
            self.path_input.setText(path)
            self.status_bar.showMessage("Download complete")
            
            # Auto-switch to generation tab
            self.tabs.setCurrentIndex(1)
        else:
            self.download_status.setText(f"Download failed: {path}")
            self.status_bar.showMessage("Download failed")
    
    def load_model(self):
        model_path = self.path_input.text()
        if not model_path or not os.path.exists(model_path):
            self.show_error_message("Error", "Please select a valid model path")
            return
        
        self.model_path = model_path
        self.load_status.setText(f"Model loaded: {os.path.basename(model_path)}")
        self.generation_status.setText("Ready to generate")
        self.generate_button.setEnabled(True)
        
        # Auto-switch to generation tab
        self.tabs.setCurrentIndex(1)
    
    def generate_images(self):
        if not self.model_path:
            self.show_error_message("Error", "Please load a model first")
            return
        
        # Get prompts
        prompts_text = self.prompt_input.toPlainText().strip()
        if not prompts_text:
            self.show_error_message("Error", "Please enter at least one prompt")
            return
        
        prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
        
        # Disable button during generation
        self.generate_button.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Prepare generation config
        config = {
            "prompts": prompts,
            "negative_prompt": self.neg_prompt_input.text(),
            "steps": self.steps_input.value(),
            "guidance_scale": self.guidance_input.value(),
            "height": self.height_input.value(),
            "width": self.width_input.value(),
            "seed": self.seed_input.value(),
            "use_cuda": self.device_checkbox.isChecked(),
            "output_dir": self.output_input.text()
        }
        
        # Start generation thread
        self.generate_thread = GenerateImagesThread(
            model_path=self.model_path,
            config=config
        )
        self.generate_thread.progress_signal.connect(self.update_generation_status)
        self.generate_thread.image_signal.connect(self.add_generated_image)
        self.generate_thread.finished_signal.connect(self.generation_completed)
        self.generate_thread.start()
    
    def update_generation_status(self, message):
        self.generation_status.setText(message)
        self.status_bar.showMessage(message)
    
    def add_generated_image(self, qimage, filepath):
        # Store the image path for the gallery
        self.images_generated.append(filepath)
        
        # Add to gallery if we're on the gallery tab
        self.refresh_gallery()
        
        # Switch to gallery tab to show the image
        self.tabs.setCurrentIndex(2)
        
        # Select the newly generated image in the list
        items = self.image_list.findItems(os.path.basename(filepath), Qt.MatchFlag.MatchExactly)
        if items:
            self.image_list.setCurrentItem(items[0])
            self.show_selected_image(items[0])
    
    def generation_completed(self):
        # Re-enable button
        self.generate_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status
        self.status_bar.showMessage("Generation complete")
        
        # Switch to gallery tab
        self.tabs.setCurrentIndex(2)
    
    def refresh_gallery(self):
        # Clear the list
        self.image_list.clear()
        
        # Get all PNG files in the output directory
        output_dir = self.output_input.text()
        if not os.path.exists(output_dir):
            return
        
        image_files = []
        for file in os.listdir(output_dir):
            if file.lower().endswith(".png"):
                image_files.append(file)
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)), reverse=True)
        
        # Add to list
        for image_file in image_files:
            self.image_list.addItem(image_file)
    
    def show_selected_image(self, item):
        if not item:
            return
        
        # Get image path
        image_filename = item.text()
        image_path = os.path.join(self.output_input.text(), image_filename)
        
        # Load and display the image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale pixmap to fit the label while maintaining aspect ratio
            self.image_preview.setPixmap(pixmap.scaled(
                self.image_preview.width(), 
                self.image_preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Load metadata if available
            metadata_path = image_path.replace(".png", ".json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Format metadata as text
                    metadata_text = "Image Metadata:\n\n"
                    for key, value in metadata.items():
                        metadata_text += f"{key}: {value}\n"
                    
                    self.image_metadata.setText(metadata_text)
                except Exception as e:
                    self.image_metadata.setText(f"Error loading metadata: {str(e)}")
            else:
                self.image_metadata.setText("No metadata found for this image")
        else:
            self.image_preview.setText("Error loading image")
            self.image_metadata.setText("No metadata available")
    
    def show_error_message(self, title, message):
        QMessageBox.critical(self, title, message)
    
    def resizeEvent(self, event):
        # Update image preview when window is resized
        if hasattr(self, 'image_list') and self.image_list.currentItem():
            self.show_selected_image(self.image_list.currentItem())
        
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        # Clean up any running threads
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.terminate()
            self.download_thread.wait()
            
        if self.generate_thread and self.generate_thread.isRunning():
            self.generate_thread.terminate()
            self.generate_thread.wait()
            
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    
    # Create and show the application window
    main_window = FluxImageGenerator()
    main_window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()