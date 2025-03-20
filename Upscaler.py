import sys
import os
import torch
from PIL import Image, ImageQt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QFileDialog, QSlider, QVBoxLayout, QHBoxLayout, 
                           QWidget, QGroupBox, QSpinBox, QDoubleSpinBox, 
                           QProgressBar, QComboBox, QMessageBox, QLineEdit,
                           QCheckBox, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor, QPalette

# Check if required packages are installed, if not show instructions
try:
    from diffusers.utils import load_image
    from diffusers import FluxControlNetModel
    from diffusers.pipelines import FluxControlNetPipeline
    import huggingface_hub
except ImportError:
    print("Required packages not found. Please install with:")
    print("pip install torch diffusers transformers accelerate huggingface_hub")
    sys.exit(1)

class HFTokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face Token Setup")
        self.resize(400, 180)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Add info label
        info_label = QLabel(
            "Enter your Hugging Face access token to download models.\n"
            "Get your token from: https://huggingface.co/settings/tokens"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Token input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Token:"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        token_layout.addWidget(self.token_input)
        layout.addLayout(token_layout)
        
        # Remember token option
        self.remember_token = QCheckBox("Remember token")
        self.remember_token.setChecked(True)
        layout.addWidget(self.remember_token)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def getToken(self):
        return self.token_input.text()
        
    def rememberToken(self):
        return self.remember_token.isChecked()

class UpscalerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(Image.Image)
    error = pyqtSignal(str)
    
    def __init__(self, input_image, scale_factor, prompt, guidance_scale, 
                 conditioning_scale, steps, device, hf_token=None):
        super().__init__()
        self.input_image = input_image
        self.scale_factor = scale_factor
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.conditioning_scale = conditioning_scale
        self.steps = steps
        self.device = device
        self.hf_token = hf_token
        
    def run(self):
        try:
            # Set Hugging Face token if provided
            if self.hf_token:
                huggingface_hub.login(token=self.hf_token)
            
            # Load models
            self.progress.emit(10)
            
            controlnet = FluxControlNetModel.from_pretrained(
                "jasperai/Flux.1-dev-Controlnet-Upscaler",
                torch_dtype=torch.bfloat16,
                use_auth_token=self.hf_token
            )
            
            self.progress.emit(30)
            
            pipe = FluxControlNetPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                controlnet=controlnet,
                torch_dtype=torch.bfloat16,
                use_auth_token=self.hf_token
            )
            
            self.progress.emit(50)
            
            pipe.to(self.device)
            
            # Prepare image
            w, h = self.input_image.size
            control_image = self.input_image.resize((w * self.scale_factor, h * self.scale_factor))
            
            self.progress.emit(60)
            
            # Custom callback for progress tracking
            def callback_fn(step, timestep, latents):
                progress = 60 + int((step / self.steps) * 35)
                self.progress.emit(progress)
                return True
            
            # Run inference
            output = pipe(
                prompt=self.prompt,
                control_image=control_image,
                controlnet_conditioning_scale=self.conditioning_scale,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                height=control_image.size[1],
                width=control_image.size[0],
                callback=callback_fn,
                callback_steps=1
            ).images[0]
            
            self.progress.emit(100)
            self.finished.emit(output)
            
        except Exception as e:
            self.error.emit(str(e))

class FluxUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_image = None
        self.output_image = None
        self.hf_token = None
        self.settings = QSettings("FluxAI", "Upscaler")
        self.loadSettings()
        self.initUI()
        
    def loadSettings(self):
        # Load Hugging Face token if saved
        if self.settings.contains("hf_token"):
            self.hf_token = self.settings.value("hf_token")
        
    def saveSettings(self):
        # Save settings
        if self.hf_token and self.remember_token:
            self.settings.setValue("hf_token", self.hf_token)
        elif not self.remember_token and self.settings.contains("hf_token"):
            self.settings.remove("hf_token")
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('Flux AI Upscaler')
        self.setMinimumSize(1000, 700)
        
        # Set up the main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Set color theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2D3250;
                color: #F0ECE5;
            }
            QPushButton {
                background-color: #7077A1;
                color: #F0ECE5;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #424874;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #909090;
            }
            QLabel {
                color: #F0ECE5;
            }
            QGroupBox {
                border: 1px solid #7077A1;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #424874;
                height: 8px;
                background: #2D3250;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #7077A1;
                border: 1px solid #424874;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #424874;
                color: #F0ECE5;
                border: 1px solid #7077A1;
                border-radius: 4px;
                padding: 2px 5px;
            }
            QProgressBar {
                border: 1px solid #7077A1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #7077A1;
                width: 20px;
            }
        """)
        
        # Create header
        header_layout = QHBoxLayout()
        app_title = QLabel("Flux AI Image Upscaler")
        app_title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(app_title)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Create image display area
        image_layout = QHBoxLayout()
        
        # Input image area
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout(input_group)
        
        self.input_display = QLabel("No image loaded")
        self.input_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_display.setMinimumSize(400, 300)
        self.input_display.setStyleSheet("border: 1px dashed #7077A1;")
        
        input_buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.loadImage)
        input_buttons_layout.addWidget(self.load_button)
        
        input_layout.addWidget(self.input_display)
        input_layout.addLayout(input_buttons_layout)
        
        # Output image area
        output_group = QGroupBox("Upscaled Result")
        output_layout = QVBoxLayout(output_group)
        
        self.output_display = QLabel("Result will appear here")
        self.output_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_display.setMinimumSize(400, 300)
        self.output_display.setStyleSheet("border: 1px dashed #7077A1;")
        
        output_buttons_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.saveResult)
        self.save_button.setEnabled(False)
        output_buttons_layout.addWidget(self.save_button)
        
        output_layout.addWidget(self.output_display)
        output_layout.addLayout(output_buttons_layout)
        
        image_layout.addWidget(input_group)
        image_layout.addWidget(output_group)
        main_layout.addLayout(image_layout)
        
        # Controls area
        controls_group = QGroupBox("Upscaling Settings")
        controls_layout = QVBoxLayout(controls_group)
        
        # Settings layout
        settings_layout = QHBoxLayout()
        
        # First column
        col1_layout = QVBoxLayout()
        
        # Scale factor
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale Factor:"))
        self.scale_factor = QSpinBox()
        self.scale_factor.setMinimum(2)
        self.scale_factor.setMaximum(8)
        self.scale_factor.setValue(4)
        scale_layout.addWidget(self.scale_factor)
        col1_layout.addLayout(scale_layout)
        
        # Guidance scale
        guidance_layout = QHBoxLayout()
        guidance_layout.addWidget(QLabel("Guidance Scale:"))
        self.guidance_scale = QDoubleSpinBox()
        self.guidance_scale.setMinimum(1.0)
        self.guidance_scale.setMaximum(10.0)
        self.guidance_scale.setValue(3.5)
        self.guidance_scale.setSingleStep(0.1)
        guidance_layout.addWidget(self.guidance_scale)
        col1_layout.addLayout(guidance_layout)
        
        # Second column
        col2_layout = QVBoxLayout()
        
        # Conditioning scale
        conditioning_layout = QHBoxLayout()
        conditioning_layout.addWidget(QLabel("Conditioning Scale:"))
        self.conditioning_scale = QDoubleSpinBox()
        self.conditioning_scale.setMinimum(0.1)
        self.conditioning_scale.setMaximum(1.0)
        self.conditioning_scale.setValue(0.6)
        self.conditioning_scale.setSingleStep(0.05)
        conditioning_layout.addWidget(self.conditioning_scale)
        col2_layout.addLayout(conditioning_layout)
        
        # Steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Inference Steps:"))
        self.steps = QSpinBox()
        self.steps.setMinimum(10)
        self.steps.setMaximum(100)
        self.steps.setValue(28)
        steps_layout.addWidget(self.steps)
        col2_layout.addLayout(steps_layout)
        
        # Third column
        col3_layout = QVBoxLayout()
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device = QComboBox()
        # Add available CUDA devices
        self.device.addItem("cuda")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device.addItem(f"cuda:{i}")
        else:
            self.device.setEnabled(False)
            self.device.setCurrentText("No CUDA available")
        device_layout.addWidget(self.device)
        col3_layout.addLayout(device_layout)
        
        # Prompt
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt = QComboBox()
        self.prompt.setEditable(True)
        self.prompt.addItem("")
        self.prompt.addItem("HD, highly detailed, sharp focus")
        self.prompt.addItem("4K, ultra HD, crystal clear, highly detailed")
        self.prompt.addItem("Professional photo, sharp, 8K, cinema quality")
        prompt_layout.addWidget(self.prompt)
        col3_layout.addLayout(prompt_layout)
        
        settings_layout.addLayout(col1_layout)
        settings_layout.addLayout(col2_layout)
        settings_layout.addLayout(col3_layout)
        controls_layout.addLayout(settings_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        controls_layout.addLayout(progress_layout)
        
        # Process button
        button_layout = QHBoxLayout()
        
        self.token_button = QPushButton("Set HF Token")
        self.token_button.clicked.connect(self.setHFToken)
        button_layout.addWidget(self.token_button)
        
        self.process_button = QPushButton("Upscale Image")
        self.process_button.setMinimumHeight(40)
        self.process_button.clicked.connect(self.processImage)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)
        
        controls_layout.addLayout(button_layout)
        
        main_layout.addWidget(controls_group)
        
        # Status bar
        self.statusBar().showMessage('Ready')
        self.updateTokenStatus()
        
        # Center window
        self.center()
        
    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, 
                 (screen.height() - size.height()) // 2)
        
    def loadImage(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)", 
            options=options)
        
        if file_name:
            try:
                self.input_image = Image.open(file_name)
                pixmap = QPixmap(file_name)
                
                # Scale pixmap to fit the label while maintaining aspect ratio
                pixmap = pixmap.scaled(
                    self.input_display.width(), 
                    self.input_display.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.input_display.setPixmap(pixmap)
                self.process_button.setEnabled(True)
                
                # Update status and info
                w, h = self.input_image.size
                scale = self.scale_factor.value()
                self.statusBar().showMessage(f'Loaded image: {w}x{h}. Will upscale to {w*scale}x{h*scale}')
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def setHFToken(self):
        dialog = HFTokenDialog(self)
        if self.hf_token:
            dialog.token_input.setText(self.hf_token)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.hf_token = dialog.getToken()
            self.remember_token = dialog.rememberToken()
            self.saveSettings()
            self.updateTokenStatus()
    
    def updateTokenStatus(self):
        if self.hf_token:
            self.token_button.setText("HF Token ✓")
            self.token_button.setStyleSheet("background-color: #7077A1;")
        else:
            self.token_button.setText("Set HF Token")
            self.token_button.setStyleSheet("")
        
    def processImage(self):
        if self.input_image is None:
            return
            
        # Check if token is set
        if not self.hf_token:
            reply = QMessageBox.question(
                self, 
                "No Hugging Face Token", 
                "No Hugging Face token is set. This may prevent downloading models.\nDo you want to set a token now?", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.setHFToken()
                # If user canceled token setting, don't proceed
                if not self.hf_token:
                    return
            
        # Disable UI during processing
        self.process_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Get parameters
        scale_factor = self.scale_factor.value()
        prompt = self.prompt.currentText()
        guidance_scale = self.guidance_scale.value()
        conditioning_scale = self.conditioning_scale.value()
        steps = self.steps.value()
        device = self.device.currentText()
        
        # Update status
        self.statusBar().showMessage('Processing image...')
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Start upscaling in a thread
        self.thread = UpscalerThread(
            self.input_image, 
            scale_factor, 
            prompt, 
            guidance_scale,
            conditioning_scale, 
            steps,
            device,
            self.hf_token
        )
        
        self.thread.progress.connect(self.updateProgress)
        self.thread.finished.connect(self.handleResult)
        self.thread.error.connect(self.handleError)
        self.thread.start()
            
    def updateProgress(self, value):
        self.progress_bar.setValue(value)
        
    def handleResult(self, result):
        self.output_image = result
        
        # Convert PIL image to QPixmap
        qt_image = ImageQt.ImageQt(result)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit the label
        pixmap = pixmap.scaled(
            self.output_display.width(), 
            self.output_display.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.output_display.setPixmap(pixmap)
        
        # Re-enable UI
        self.process_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Update status
        w, h = result.size
        self.statusBar().showMessage(f'Upscaling complete! Result size: {w}x{h}')
        
    def handleError(self, error_msg):
        # Re-enable UI
        self.process_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # Show error
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")
        self.statusBar().showMessage('Error during processing')
        
    def saveResult(self):
        if self.output_image is None:
            return
            
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", 
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)", 
            options=options)
            
        if file_name:
            try:
                # Add extension if missing
                if not any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    file_name += '.png'
                    
                self.output_image.save(file_name)
                self.statusBar().showMessage(f'Saved to {file_name}')
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
        
def main():
    app = QApplication(sys.argv)
    ex = FluxUpscalerApp()
    ex.show()
    sys.exit(app.exec())
    
if __name__ == '__main__':
    main()