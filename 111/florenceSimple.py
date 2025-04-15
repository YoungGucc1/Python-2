import sys
import os
import subprocess
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QIcon # Optional: for window icon

# --- Dependency Check and Installation ---
try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'Pillow'], check=True)
    from PIL import Image

try:
    import torch
    import transformers
except ImportError:
    print("torch or transformers not found. Installing...")
    # Install PyTorch first, then transformers
    # This might need specific versions depending on CUDA, adjust if necessary
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers'], check=True)
    import torch
    import transformers

# Optional: Try installing flash-attn, but don't make it mandatory
# It might fail on systems without compatible GPUs or build tools
# Note: 'spaces' library is specific to Hugging Face Spaces, not needed here.
try:
    print("Attempting to install flash-attn (optional)...")
    subprocess.run(
        f'{sys.executable} -m pip install flash-attn --no-build-isolation',
        env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE", **os.environ},
        shell=True,
        check=False, # Don't fail the whole script if this fails
        capture_output=True,
        text=True
    )
    print("flash-attn installation attempt finished.")
except Exception as e:
    print(f"Could not install or run flash-attn: {e}")

from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration ---
# Model is cached at C:\Users\PC\.cache\huggingface\hub
HUGGINGFACE_CACHE = os.path.expanduser("C:\\Users\\PC\\.cache\\huggingface\\hub")
MODEL_ID = "microsoft/Florence-2-large-ft"  # Original model ID
MODEL_CACHE_PATH = os.path.join(HUGGINGFACE_CACHE, "models--microsoft--Florence-2-large-ft")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Looking for cached model at: {MODEL_CACHE_PATH}")

# --- Model Loading ---
# Global variables for model and processor (or manage within the App class)
# We load them lazily or in a separate thread to avoid blocking GUI startup.
model = None
processor = None
model_load_lock = threading.Lock()
model_loaded_event = threading.Event()

def load_model_and_processor():
    """Loads the model and processor. To be run in a separate thread."""
    global model, processor
    try:
        print(f"Starting model loading from: {MODEL_CACHE_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Checking if cache directory exists: {os.path.exists(MODEL_CACHE_PATH)}")
        
        if os.path.exists(MODEL_CACHE_PATH):
            print(f"Cache directory found! Contents of {MODEL_CACHE_PATH}:")
            snapshots_dir = os.path.join(MODEL_CACHE_PATH, "snapshots")
            if os.path.exists(snapshots_dir):
                print(f"Snapshot directory exists: {snapshots_dir}")
                snapshot_folders = os.listdir(snapshots_dir)
                if snapshot_folders:
                    print(f"Found snapshots: {snapshot_folders}")
                    # Get the latest snapshot (usually there's just one)
                    latest_snapshot = os.path.join(snapshots_dir, snapshot_folders[-1])
                    print(f"Using snapshot: {latest_snapshot}")
                    
                    # Load model with appropriate dtype if using CPU or CUDA
                    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                    print("Loading model from snapshot...")
                    model = AutoModelForCausalLM.from_pretrained(
                        latest_snapshot,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        local_files_only=True
                    ).to(DEVICE).eval()
                    
                    print("Loading processor from snapshot...")
                    processor = AutoProcessor.from_pretrained(
                        latest_snapshot, 
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    print(f"Model and processor loaded successfully from cache. Model is None? {model is None}, Processor is None? {processor is None}")
                    model_loaded_event.set()
                    print("Set model_loaded_event to True")
                    return
                else:
                    print("No snapshots found in the snapshots directory.")
            else:
                print(f"No snapshots directory found in {MODEL_CACHE_PATH}")
        
        # If we reach here, we couldn't load from cache directly, try with model ID
        # This will still use the cache but through Hugging Face's API
        print(f"Falling back to loading with original model ID: {MODEL_ID}")
        
        # Load model with appropriate dtype if using CPU or CUDA
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        print("Attempting to load model with model ID...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=HUGGINGFACE_CACHE
        ).to(DEVICE).eval()

        print(f"Loading processor with model ID {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            cache_dir=HUGGINGFACE_CACHE
        )
        print(f"Model and processor loaded successfully. Model is None? {model is None}, Processor is None? {processor is None}")
        model_loaded_event.set()
        print("Set model_loaded_event to True")
    except Exception as e:
        print(f"Error loading model/processor: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        model = None # Ensure model is None if loading failed
        processor = None
        print("Model or processor is None due to error. Not setting model_loaded_event.")

# --- Worker Thread for Inference ---
class InferenceWorker(QObject):
    """Handles model inference in a separate thread."""
    progress = pyqtSignal(int, int)  # current_index, total_count
    result_ready = pyqtSignal(int, str, str) # row_index, status, answer/error
    finished = pyqtSignal()

    def __init__(self, image_paths, question):
        super().__init__()
        self.image_paths = image_paths
        self.question = question
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not model_loaded_event.is_set():
            print("Waiting for model to load...")
            model_loaded_event.wait() # Wait until the model is loaded
            if not model or not processor:
                print("Model not loaded correctly. Aborting inference.")
                self.result_ready.emit(0, "Error", "Model failed to load.")
                self.finished.emit()
                return

        print("Starting inference batch...")
        task_prompt = '<DocVQA>'
        total_images = len(self.image_paths)

        for i, img_path in enumerate(self.image_paths):
            if not self._is_running:
                print("Inference stopped by user.")
                break

            self.progress.emit(i, total_images)
            try:
                image = Image.open(img_path).convert("RGB") # Ensure RGB
                prompt = task_prompt + self.question

                # Process inputs
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
                # Ensure inputs match model dtype if needed (especially for float16 on CUDA)
                if DEVICE == "cuda" and model.dtype == torch.float16:
                    inputs = inputs.to(torch.float16)

                # Generate
                with torch.no_grad(): # Important for inference
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        early_stopping=False,
                        do_sample=False,
                        num_beams=3,
                    )

                # Decode and post-process
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                # The post_process_generation function expects the task *without* the question
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=task_prompt, # Use only the task marker
                    image_size=(image.width, image.height)
                )

                # Clean up the answer string
                answer = parsed_answer.get(task_prompt, "Error: Task key not found").replace("<pad>", "").strip()
                self.result_ready.emit(i, "Success", answer)

            except FileNotFoundError:
                self.result_ready.emit(i, "Error", f"File not found: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                self.result_ready.emit(i, "Error", str(e))

        self.progress.emit(total_images, total_images) # Signal completion
        self.finished.emit()
        print("Inference batch finished.")

# --- Main Application Window ---
class DocVQAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Florence-2 DocVQA Batch Processor (HF Cache: {os.path.basename(HUGGINGFACE_CACHE)})")
        self.setGeometry(100, 100, 900, 700) # x, y, width, height
        # Optional: Set an icon (create an icon file like 'icon.png')
        # self.setWindowIcon(QIcon('icon.png'))

        self.selected_files = []
        self.worker_thread = None
        self.inference_worker = None
        
        # Create force enable button but don't add it to layout yet
        self.force_enable_button = QPushButton("Force Enable Processing")
        self.force_enable_button.clicked.connect(self.force_enable_processing)
        self.force_enable_button.setVisible(False)
        self.force_enable_timer = QTimer(self)
        self.force_enable_timer.timeout.connect(self.check_show_force_enable)
        self.force_enable_timer.start(15000)  # Show after 15 seconds

        # Start loading the model in a background thread
        self.model_loader_thread = threading.Thread(target=load_model_and_processor, daemon=True)
        self.model_loader_thread.start()

        self._init_ui()
        self._apply_styles() # Apply custom styling

        # Initial status
        self.status_label.setText(f"Loading model from HF cache: {MODEL_CACHE_PATH}...")
        self.start_button.setEnabled(False) # Disable until model is loaded

        # Use a timer to check if the model has loaded, then enable the button
        self.check_model_timer = QTimer(self)
        self.check_model_timer.timeout.connect(self._check_model_loaded)
        self.check_model_timer.start(500) # Check every 500ms

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Input Area ---
        input_group = QWidget()
        input_layout = QHBoxLayout(input_group)

        # File Selection
        file_selection_layout = QVBoxLayout()
        self.select_button = QPushButton("Select Images")
        self.select_button.clicked.connect(self.select_images)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setToolTip("Selected image files for batch processing")
        file_selection_layout.addWidget(self.select_button)
        file_selection_layout.addWidget(QLabel("Selected Files:"))
        file_selection_layout.addWidget(self.file_list_widget)

        # Question Input
        question_layout = QVBoxLayout()
        question_layout.addWidget(QLabel("Question for all images:"))
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your question here")
        self.question_input.setToolTip("This question will be asked for every selected image.")
        question_layout.addWidget(self.question_input)
        question_layout.addStretch() # Pushes the input field to the top

        input_layout.addLayout(file_selection_layout, stretch=2) # Give file list more space
        input_layout.addLayout(question_layout, stretch=1)
        main_layout.addWidget(input_group)


        # --- Control Area ---
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Batch Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False) # Initially disabled

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False) # Hide until processing starts

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        # The force_enable_button will be added later dynamically if needed
        control_layout.addWidget(self.progress_bar, stretch=1) # Give progress bar more space
        main_layout.addLayout(control_layout)

        # --- Output Area ---
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Image Path", "Question", "Status", "Answer"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive) # Allow resizing path
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.results_table.setAlternatingRowColors(True) # Improves readability
        main_layout.addWidget(QLabel("Results:"))
        main_layout.addWidget(self.results_table)

        # --- Status Bar ---
        self.status_label = QLabel("Ready.")
        main_layout.addWidget(self.status_label)

    def _apply_styles(self):
        """Applies QSS styling for a modern look."""
        style_sheet = """
            QMainWindow {
                background-color: #f0f0f0; /* Light gray background */
            }
            QWidget { /* Default font for the application */
                font-size: 10pt;
            }
            QPushButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 11pt;
                margin: 4px 2px;
                border-radius: 5px; /* Rounded corners */
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049; /* Darker Green */
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            #stop_button { /* Specific style for stop button */
                 background-color: #f44336; /* Red */
            }
            #stop_button:hover {
                 background-color: #da190b;
            }
            #stop_button:pressed {
                 background-color: #b61f14;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 10pt;
            }
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QTableWidget {
                border: 1px solid #ccc;
                gridline-color: #e0e0e0; /* Lighter grid lines */
                background-color: white;
                alternate-background-color: #f8f8f8; /* Slightly off-white for alternating rows */
            }
            QHeaderView::section {
                background-color: #e0e0e0; /* Header background */
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-size: 10pt;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 20px; /* Make progress bar slightly taller */
            }
            QProgressBar::chunk {
                background-color: #4CAF50; /* Green progress */
                width: 10px; /* Width of the progress segments */
                 margin: 0.5px;
            }
            QLabel {
                font-size: 10pt;
                padding: 2px;
            }
        """
        # Add object names for specific styling
        self.stop_button.setObjectName("stop_button")

        self.setStyleSheet(style_sheet)


    def _check_model_loaded(self):
        """Checks if the model has finished loading."""
        print(f"Checking if model loaded. Event set: {model_loaded_event.is_set()}, Model exists: {model is not None}, Processor exists: {processor is not None}")
        
        if model_loaded_event.is_set():
            self.check_model_timer.stop() # Stop checking
            if model is not None and processor is not None:
                self.status_label.setText("Model loaded. Ready to process.")
                self.start_button.setEnabled(True)
                print("Model and processor loaded successfully. Start button enabled.")
            else:
                self.status_label.setText("Error: Model failed to load. Check console.")
                print("ERROR: Model or processor is None despite event being set!")
                QMessageBox.critical(self, "Model Load Error", "Failed to load the AI model. Please check the console output for details and ensure dependencies are installed correctly.")
                
                # Add a force enable button after 10 seconds of waiting
                if not hasattr(self, '_force_enable_attempts'):
                    self._force_enable_attempts = 0
                
                self._force_enable_attempts += 1
                if self._force_enable_attempts >= 20:  # After 10 seconds (20 * 500ms)
                    print("Force enabling start button after timeout...")
                    self.start_button.setEnabled(True)
                    self.status_label.setText("WARNING: Force enabled processing, model may not be fully loaded.")
                    self.check_model_timer.stop()
        else:
            # Update status with more information
            if not hasattr(self, '_loading_time'):
                self._loading_time = 0
            
            self._loading_time += 0.5  # 500ms interval
            if self._loading_time % 5 == 0:  # Every 5 seconds
                self.status_label.setText(f"Still loading model... ({int(self._loading_time)}s)")
                print(f"Still waiting for model to load after {int(self._loading_time)} seconds.")
                
                # After 30 seconds, add option to force enable
                if self._loading_time >= 30:
                    print("Loading is taking a long time. Enabling start button...")
                    self.start_button.setEnabled(True)
                    self.status_label.setText("WARNING: Load taking too long, try processing but expect errors.")
                    self.check_model_timer.stop()


    def select_images(self):
        """Opens a dialog to select multiple image files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "", # Start directory (empty means default)
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)" # Filter
        )
        if files:
            self.selected_files = files
            self.file_list_widget.clear()
            self.file_list_widget.addItems([Path(f).name for f in files]) # Show only filenames
            print(f"Selected {len(files)} files.")
            self.status_label.setText(f"{len(files)} files selected.")

    def start_processing(self):
        """Starts the batch inference process."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select image files first.")
            return
        question = self.question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "No Question", "Please enter a question.")
            return
            
        # Check if model is ready
        if not model_loaded_event.is_set():
            response = QMessageBox.question(
                self, 
                "Model Not Ready", 
                "The model is still loading. Do you want to force start processing anyway?\n\nWARNING: This may cause errors.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.Yes:
                model_loaded_event.set()  # Force set the event
                print("User chose to force start processing.")
            else:
                return
                
        # Even if user forced start, we need to check if model/processor exists
        if model is None or processor is None:
            QMessageBox.critical(
                self, 
                "Model Not Available", 
                "The model or processor is not available. Processing cannot start.\n\nPlease check the console for error details."
            )
            return


        # Clear previous results and prepare table
        self.results_table.setRowCount(len(self.selected_files))
        for i, f_path in enumerate(self.selected_files):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(Path(f_path)))) # Full path in table item
            self.results_table.setItem(i, 1, QTableWidgetItem(question))
            self.results_table.setItem(i, 2, QTableWidgetItem("Pending"))
            self.results_table.setItem(i, 3, QTableWidgetItem("")) # Placeholder for answer

        # Setup and start worker thread
        self.worker_thread = QThread()
        self.inference_worker = InferenceWorker(self.selected_files, question)
        self.inference_worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.inference_worker.run)
        self.inference_worker.progress.connect(self.update_progress)
        self.inference_worker.result_ready.connect(self.update_result)
        self.inference_worker.finished.connect(self.on_processing_finished)
        # Ensure thread cleanup
        self.inference_worker.finished.connect(self.worker_thread.quit)
        self.inference_worker.finished.connect(self.inference_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Update UI state
        self.start_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.selected_files))
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")

        self.worker_thread.start()

    def stop_processing(self):
        """Requests the worker thread to stop."""
        if self.inference_worker:
            self.inference_worker.stop()
            self.status_label.setText("Stopping...")
            self.stop_button.setEnabled(False) # Prevent multiple clicks

    def update_progress(self, current_index, total_count):
        """Updates the progress bar and status."""
        self.progress_bar.setValue(current_index)
        if current_index < total_count:
           self.status_label.setText(f"Processing image {current_index + 1} of {total_count}...")
           # Update status in table for the *next* item being processed
           if current_index < self.results_table.rowCount():
               self.results_table.setItem(current_index, 2, QTableWidgetItem("Processing"))


    def update_result(self, row_index, status, answer_or_error):
        """Updates a row in the results table."""
        if row_index < self.results_table.rowCount():
            status_item = QTableWidgetItem(status)
            # Optional: Color code status
            if status == "Error":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "Success":
                 status_item.setForeground(Qt.GlobalColor.darkGreen)

            self.results_table.setItem(row_index, 2, status_item)
            self.results_table.setItem(row_index, 3, QTableWidgetItem(answer_or_error))

    def on_processing_finished(self):
        """Called when the worker thread completes."""
        self.status_label.setText("Batch processing finished.")
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum()) # Ensure it shows 100%
        # self.progress_bar.setVisible(False) # Optionally hide after completion
        self.worker_thread = None # Clean up reference
        self.inference_worker = None

    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_processing() # Attempt to stop thread if running
        if self.worker_thread and self.worker_thread.isRunning():
             print("Waiting for worker thread to finish...")
             # self.worker_thread.quit() # Request quit
             self.worker_thread.wait(2000) # Wait max 2 seconds
        event.accept()

    def check_show_force_enable(self):
        """Check if we should show the force enable button."""
        if not model_loaded_event.is_set():
            print("Model loading taking too long. Showing force enable button.")
            # Add the force enable button to the layout
            control_layout = None
            for i in range(self.centralWidget().layout().count()):
                item = self.centralWidget().layout().itemAt(i)
                if isinstance(item, QHBoxLayout) and item.indexOf(self.start_button) != -1:
                    control_layout = item
                    break
                
            if control_layout:
                control_layout.addWidget(self.force_enable_button)
                self.force_enable_button.setVisible(True)
                self.status_label.setText("Model loading taking too long. You can force enable processing.")
                
        # Only run once
        self.force_enable_timer.stop()
            
    def force_enable_processing(self):
        """Force enable the processing button even if the model hasn't loaded."""
        print("Force enabling processing button.")
        self.start_button.setEnabled(True)
        self.force_enable_button.setEnabled(False)
        self.status_label.setText("WARNING: Processing enabled but model may not be fully loaded!")
        # Set the event so the app thinks the model is loaded
        model_loaded_event.set()


# --- Run the Application ---
if __name__ == "__main__":
    # For Qt6, high DPI scaling is enabled by default
    # The old attribute is no longer needed
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    main_window = DocVQAApp()
    main_window.show()
    sys.exit(app.exec())