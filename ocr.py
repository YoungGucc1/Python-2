import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QLineEdit, QSizePolicy, QFrame
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# --- Import necessary ML libraries ---
try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    # Assuming qwen_vl_utils.py is in the same directory or installed
    # If not, you might need to copy its functions here or adjust imports.
    # Let's define a placeholder if it's not readily available,
    # assuming the processor handles image loading from paths.
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Warning: qwen_vl_utils not found. Assuming processor handles image paths directly.")
        # Basic placeholder - might need adjustment based on how processor expects image data
        def process_vision_info(messages):
            image_inputs = []
            video_inputs = []
            for msg in messages:
                if msg['role'] == 'user':
                    for content_item in msg['content']:
                        if content_item['type'] == 'image':
                            # Assuming AutoProcessor can handle file paths directly in its call
                            image_inputs.append(content_item['image'])
                        # Handle 'video' type if needed
            return image_inputs, video_inputs # Returning list of paths/URLs

    ML_LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing ML libraries: {e}")
    print("Please install required libraries: pip install torch transformers accelerate sentencepiece Pillow")
    ML_LIBRARIES_AVAILABLE = False
    # Define dummy classes/functions if libraries are missing to allow GUI to load
    class Qwen2VLForConditionalGeneration:
        @staticmethod
        def from_pretrained(*args, **kwargs): raise ImportError("transformers not installed")
    class AutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs): raise ImportError("transformers not installed")
    def process_vision_info(*args, **kwargs): raise ImportError("transformers or qwen_vl_utils not installed")
    import time # For dummy sleep

# --- Worker Thread for ML Inference ---
class OCRWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal(object, object) # Signal to pass loaded model/processor back

    def __init__(self, image_path, prompt, model=None, processor=None):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.model = model
        self.processor = processor
        self._is_running = True

    def run(self):
        if not ML_LIBRARIES_AVAILABLE:
            self.error.emit("Required ML libraries not installed.")
            return

        try:
            # --- 1. Load Model and Processor (if not already loaded) ---
            if self.model is None or self.processor is None:
                self.status.emit("Loading model and processor (first run)...")
                model_name = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
                try:
                    # Determine device: prefer CUDA if available
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Using device: {device}")

                    # Load the model
                    # Consider adding attn_implementation="flash_attention_2" if compatible and installed
                    loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype="auto", # or torch.bfloat16 for flash_attention
                        # attn_implementation="flash_attention_2", # Uncomment if using flash attention
                        device_map="auto" # Automatically uses CUDA if available, else CPU
                    )
                    # Load the processor
                    loaded_processor = AutoProcessor.from_pretrained(model_name)

                    self.model = loaded_model
                    self.processor = loaded_processor
                    self.model_loaded.emit(self.model, self.processor) # Send back to main thread for caching
                    self.status.emit("Model and processor loaded.")

                except Exception as e:
                    self.error.emit(f"Error loading model/processor: {e}")
                    return
            else:
                 # Use cached model/processor
                 self.status.emit("Using cached model and processor.")
                 # Ensure model is on the correct device if needed (device_map="auto" should handle this)
                 # If loading manually, ensure model.to(device) is called
                 device = next(self.model.parameters()).device # Get current device from model
                 print(f"Using cached model on device: {device}")


            if not self._is_running: return # Check if stopped

            # --- 2. Prepare Input ---
            self.status.emit("Preparing input...")
            if not os.path.exists(self.image_path):
                 self.error.emit(f"Image file not found: {self.image_path}")
                 return

            messages = [
                {
                    "role": "user",
                    "content": [
                        # Use the local image path
                        {"type": "image", "image": self.image_path},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # process_vision_info should handle the local path
            image_inputs, video_inputs = process_vision_info(messages)

            if not self._is_running: return

            inputs = self.processor(
                text=[text],
                images=image_inputs, # Should contain the loaded image data or path
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to the same device as the model
            inputs = inputs.to(device)

            if not self._is_running: return

            # --- 3. Run Inference ---
            self.status.emit("Running OCR inference...")
            with torch.no_grad(): # Important for inference
                generated_ids = self.model.generate(**inputs, max_new_tokens=512) # Increased tokens
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

            if not self._is_running: return

            # --- 4. Decode Output ---
            self.status.emit("Decoding output...")
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if not self._is_running: return

            self.status.emit("Processing complete.")
            self.finished.emit(output_text[0] if output_text else "No text generated.")

        except Exception as e:
            self.error.emit(f"An error occurred during processing: {e}")
        finally:
            # Clean up GPU memory if CUDA was used and model loaded here
            # If model is cached in main window, maybe don't cleanup here
            # if device == "cuda":
            #    del inputs
            #    torch.cuda.empty_cache()
            pass # Let main thread manage cached model lifecycle

    def stop(self):
        self._is_running = False

# --- Main GUI Window ---
class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen2-VL OCR App")
        self.setGeometry(100, 100, 800, 650) # x, y, width, height

        self.ml_model = None
        self.ml_processor = None
        self.current_image_path = None
        self.ocr_worker = None

        # --- Layouts ---
        self.main_layout = QVBoxLayout(self)
        self.top_layout = QHBoxLayout() # For controls
        self.image_layout = QHBoxLayout() # For image display and output
        self.status_layout = QHBoxLayout() # For status bar elements

        # --- Widgets ---
        # Top Controls
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        self.prompt_label = QLabel("Prompt:")
        self.prompt_input = QLineEdit("Extract text from this image.") # Default OCR prompt
        self.run_button = QPushButton("Run OCR")
        self.run_button.clicked.connect(self.run_ocr)
        self.run_button.setEnabled(False) # Disabled until image is selected

        # Image Display
        self.image_label = QLabel("No image selected.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        # Output Text Area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("OCR results will appear here...")
        self.output_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Status Bar
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("padding-left: 5px;")

        # --- Arrange Widgets ---
        # Top Layout
        self.top_layout.addWidget(self.select_button)
        self.top_layout.addWidget(self.prompt_label)
        self.top_layout.addWidget(self.prompt_input, 1) # Stretchable input
        self.top_layout.addWidget(self.run_button)

        # Image/Output Layout
        self.image_layout.addWidget(self.image_label, 1) # Give image more space initially
        self.image_layout.addWidget(self.output_text, 1) # Equal space for output

        # Status Layout
        self.status_layout.addWidget(self.status_label)
        status_frame = QFrame() # Use a frame for the status bar background
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setLayout(self.status_layout)
        status_frame.setFixedHeight(30)


        # Main Layout
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.image_layout)
        # self.main_layout.addLayout(self.status_layout) # Add status bar at the bottom
        self.main_layout.addWidget(status_frame)


        # Apply Dark Theme Stylesheet
        self.set_dark_theme()

        # Show warning if ML libs are missing
        if not ML_LIBRARIES_AVAILABLE:
            self.status_label.setText("Status: ML Libraries Missing! Install torch, transformers, etc.")
            self.run_button.setEnabled(False)
            self.select_button.setEnabled(False) # Maybe allow selecting image?


    def set_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #555555;
                color: #f0f0f0;
                border: 1px solid #666666;
                padding: 5px 10px;
                border-radius: 3px;
                min-height: 20px; /* Ensure button height */
            }
            QPushButton:hover {
                background-color: #6a6a6a;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QLineEdit, QTextEdit {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QLabel {
                color: #f0f0f0;
                padding-right: 5px; /* Add spacing for labels like "Prompt:" */
            }
            QFrame { /* Style frame background and border */
                background-color: #333333; /* Slightly different status bar background */
                border: none;
                border-top: 1px solid #444444; /* Separator line */
            }
            # /* Style specific widgets if needed by setting objectName */
            # QLabel#image_label { /* Example if you set object name */
            #     border: 1px dashed #555555;
            # }
        """)
        # Adjust font for better readability
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.output_text.setFont(font) # Ensure text edit uses it too
        self.prompt_input.setFont(font)

    def select_image(self):
        file_dialog = QFileDialog(self)
        # Use PNG and JPG/JPEG as common image types
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.image_label.size() * 0.95, # Scale slightly smaller than label
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.run_button.setEnabled(True)
            self.output_text.setPlainText("") # Clear previous results
            self.status_label.setText(f"Status: Loaded '{os.path.basename(file_path)}'")
        else:
            # Keep previous state if dialog is cancelled
            # self.current_image_path = None
            # self.image_label.setText("No image selected.")
            # self.run_button.setEnabled(False)
            pass


    def run_ocr(self):
        if not self.current_image_path:
            self.status_label.setText("Status: Error - No image selected.")
            return
        if not ML_LIBRARIES_AVAILABLE:
             self.status_label.setText("Status: Error - ML Libraries Missing!")
             return

        # Disable button, clear output
        self.run_button.setEnabled(False)
        self.select_button.setEnabled(False) # Also disable select during processing
        self.output_text.setPlainText("")
        prompt = self.prompt_input.text()
        self.status_label.setText("Status: Starting OCR...")

        # Stop previous worker if it's still running (shouldn't happen with button disabling)
        if self.ocr_worker and self.ocr_worker.isRunning():
            print("Warning: Previous OCR task still running. Attempting to stop.")
            self.ocr_worker.stop() # Signal the worker to stop
            self.ocr_worker.wait(1000) # Wait a bit for it to finish


        # Create and start the worker thread
        self.ocr_worker = OCRWorker(self.current_image_path, prompt, self.ml_model, self.ml_processor)
        self.ocr_worker.status.connect(self.update_status)
        self.ocr_worker.finished.connect(self.on_ocr_finished)
        self.ocr_worker.error.connect(self.on_ocr_error)
        self.ocr_worker.model_loaded.connect(self.cache_model_processor) # Connect caching signal
        self.ocr_worker.start()

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def cache_model_processor(self, model, processor):
        """Stores the loaded model and processor in the main window."""
        self.ml_model = model
        self.ml_processor = processor
        print("Model and processor cached in main window.")

    def on_ocr_finished(self, result):
        self.output_text.setPlainText(result)
        self.status_label.setText("Status: OCR Finished.")
        self.run_button.setEnabled(True) # Re-enable button
        self.select_button.setEnabled(True)
        self.ocr_worker = None # Clear worker reference

    def on_ocr_error(self, error_message):
        self.output_text.setPlainText(f"Error:\n{error_message}")
        self.status_label.setText("Status: Error occurred.")
        self.run_button.setEnabled(True) # Re-enable button
        self.select_button.setEnabled(True)
        self.ocr_worker = None # Clear worker reference

    # Ensure image resizes correctly if window resizes
    def resizeEvent(self, event):
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            scaled_pixmap = pixmap.scaled(self.image_label.size() * 0.95,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    # Clean up worker thread on close
    def closeEvent(self, event):
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.stop()
            self.ocr_worker.wait() # Wait for thread to finish
        # Optional: Explicitly delete model to free memory if needed,
        # but Python's GC should handle it eventually.
        # del self.ml_model
        # del self.ml_processor
        # if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set global font size slightly larger if desired
    # font = QFont()
    # font.setPointSize(11)
    # app.setFont(font)

    window = OCRApp()
    window.show()
    sys.exit(app.exec())