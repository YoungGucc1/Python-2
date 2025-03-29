# --- START OF FILE object-detect-pyqt6.py ---

import sys
import os
import json
import time
import shutil
import cv2
from ultralytics import YOLO
from PIL import Image # For color space conversion if needed, though OpenCV handles most

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QMutex, QWaitCondition
from PyQt6.QtGui import QPixmap, QImage, QFont

# --- Constants ---
DEFAULT_MODEL = "yolov8n.pt"  # Default nano model
RESULTS_FOLDER_NAME = "detection_results_annotated"
JSON_RESULTS_FILENAME = "detection_results.json"

# --- Worker Thread for Processing ---
class Worker(QObject):
    """Handles the background image processing"""
    progressUpdated = pyqtSignal(int, int)  # current_value, max_value
    statusUpdated = pyqtSignal(str)
    imageProcessed = pyqtSignal(str, object) # image_path, cv2_image_with_detections (or None)
    processingFinished = pyqtSignal(dict)   # results_data
    errorOccurred = pyqtSignal(str)

    def __init__(self, model_path, folder_path, files):
        super().__init__()
        self.model_path = model_path
        self.folder_path = folder_path
        self.files = files
        self.model = None
        self._is_running = True
        self._is_paused = False
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()

    def run(self):
        """Main processing loop"""
        results_data = {}
        total_files = len(self.files)

        try:
            self.statusUpdated.emit(f"Loading model: {os.path.basename(self.model_path)}...")
            self.model = YOLO(self.model_path)
            self.statusUpdated.emit(f"Model loaded. Starting detection in: {self.folder_path}")
        except Exception as e:
            self.errorOccurred.emit(f"Error loading YOLO model ({self.model_path}): {e}")
            return # Stop processing if model fails

        for idx, filename in enumerate(self.files):
            self._mutex.lock()
            while self._is_paused:
                self._pause_cond.wait(self._mutex) # Wait until resumed

            should_stop = not self._is_running
            self._mutex.unlock()

            if should_stop:
                self.statusUpdated.emit("Processing stopped by user.")
                break # Exit loop if stopped

            file_path = os.path.join(self.folder_path, filename)
            self.statusUpdated.emit(f"Processing: {filename} ({idx + 1}/{total_files})")

            try:
                # Read image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Warning: Could not read image {filename}. Skipping.")
                    results_data[filename] = [] # Store empty result for this file
                    self.imageProcessed.emit(file_path, None) # Signal that it was processed (even if failed)
                    self.progressUpdated.emit(idx + 1, total_files)
                    continue

                # Perform detection
                # Use stream=True for potentially better memory management with many images
                results = self.model(image, verbose=False) # verbose=False reduces console output

                # Process and store results
                detections = []
                annotated_image = image.copy() # Make a copy for drawing

                if results and results[0].boxes:
                    for r in results[0].boxes:
                        class_id = int(r.cls.item())
                        confidence = float(r.conf.item())
                        box = [int(x) for x in r.xyxy[0].tolist()] # Use int for drawing
                        class_name = self.model.names.get(class_id, f"ID:{class_id}") # Handle unknown IDs

                        detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "box": box # Store int box for consistency
                        })

                        # Draw on the annotated_image
                        color = (0, 255, 0) # Green for all detections for simplicity now
                        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        label_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 15
                        cv2.putText(annotated_image, label, (box[0], label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                results_data[filename] = detections
                self.imageProcessed.emit(file_path, annotated_image) # Send annotated image
                self.progressUpdated.emit(idx + 1, total_files)

            except Exception as e:
                self.errorOccurred.emit(f"Error processing {filename}: {e}")
                results_data[filename] = [] # Store empty result on error
                self.imageProcessed.emit(file_path, None) # Still signal progress
                self.progressUpdated.emit(idx + 1, total_files)
                # Optionally decide whether to continue or stop on error

            # Brief sleep to allow GUI updates and reduce CPU max-out
            # time.sleep(0.01)

        if self._is_running: # Only emit finished if not stopped
             self.statusUpdated.emit("Processing finished.")
             self.processingFinished.emit(results_data)
        # Worker automatically finishes here

    def stop(self):
        self._mutex.lock()
        self._is_running = False
        if self._is_paused: # If paused, wake it up to check the stop flag
            self._is_paused = False
            self._pause_cond.wakeAll()
        self._mutex.unlock()

    def pause(self):
        self._mutex.lock()
        self._is_paused = True
        self.statusUpdated.emit("Processing paused.")
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        if self._is_paused:
            self._is_paused = False
            self.statusUpdated.emit("Processing resumed.")
            self._pause_cond.wakeAll() # Wake up the waiting thread
        self._mutex.unlock()

# --- Main Application Window ---
class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection with YOLOv8 (PyQt6)")
        self.setGeometry(100, 100, 1024, 768) # x, y, width, height

        # --- State Variables ---
        self.folder_path = ""
        self.model_path = DEFAULT_MODEL
        self.result_data = {}
        self.current_files = []
        self.worker_thread = None
        self.worker = None
        self.is_processing = False

        # --- UI Setup ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self._create_controls_section()
        self._create_image_display_section()
        self._create_status_section()
        self._apply_styles() # Apply custom styles

        self._update_button_states() # Initial button state

    def _create_controls_section(self):
        controls_layout = QVBoxLayout()

        # Model Selection
        model_layout = QHBoxLayout()
        model_label = QLabel("YOLO Model (.pt):")
        self.model_path_edit = QLineEdit(self.model_path)
        self.model_path_edit.setReadOnly(True)
        select_model_btn = QPushButton("Browse...")
        select_model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(select_model_btn)
        controls_layout.addLayout(model_layout)

        # Folder Selection
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Image Folder:")
        self.folder_path_edit = QLineEdit("No folder selected")
        self.folder_path_edit.setReadOnly(True)
        select_folder_btn = QPushButton("Select Folder...")
        select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(select_folder_btn)
        controls_layout.addLayout(folder_layout)

        # Action Buttons
        action_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Detection")
        self.start_button.setObjectName("StartButton") # For styling
        self.start_button.clicked.connect(self.start_detection)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton") # For styling
        self.stop_button.clicked.connect(self.stop_detection)

        self.draw_save_button = QPushButton("Draw & Save Results")
        self.draw_save_button.clicked.connect(self.draw_and_save_images)

        action_buttons_layout.addWidget(self.start_button)
        action_buttons_layout.addWidget(self.pause_button)
        action_buttons_layout.addWidget(self.stop_button)
        action_buttons_layout.addStretch() # Push draw button to the right
        action_buttons_layout.addWidget(self.draw_save_button)
        controls_layout.addLayout(action_buttons_layout)

        self.main_layout.addLayout(controls_layout)

    def _create_image_display_section(self):
        self.image_label = QLabel("Select a folder and start detection to see images here.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow shrinking/expanding
        self.image_label.setMinimumSize(400, 300) # Minimum sensible size
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.main_layout.addWidget(self.image_label, 1) # Give image label stretchy space (proportion 1)

    def _create_status_section(self):
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Idle.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        self.main_layout.addLayout(status_layout)

    def _apply_styles(self):
        """Apply some basic styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e8effa; /* Light blue background */
            }
            QPushButton {
                padding: 8px 15px;
                font-size: 10pt;
                background-color: #5dadec; /* Nice blue */
                color: white;
                border: none;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4a8fcc; /* Darker blue on hover */
            }
            QPushButton:disabled {
                background-color: #c0c0c0; /* Gray when disabled */
                color: #666666;
            }
            QPushButton#StartButton { /* Specific style for Start */
                font-weight: bold;
                 background-color: #4CAF50; /* Green */
            }
             QPushButton#StartButton:hover {
                 background-color: #45a049;
             }
            QPushButton#StopButton { /* Specific style for Stop */
                background-color: #f44336; /* Red */
            }
            QPushButton#StopButton:hover {
                background-color: #da190b;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QLabel {
                font-size: 10pt;
                padding: 5px 0; /* Add some vertical padding */
            }
            QProgressBar {
                text-align: center;
                height: 25px;
                font-weight: bold;
                color: #333; /* Darker text for visibility */
            }
            QProgressBar::chunk {
                background-color: #5dadec; /* Blue progress */
                border-radius: 3px;
            }
        """)

    # --- UI Action Methods ---
    def select_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if dir_path:
            self.folder_path = dir_path
            self.folder_path_edit.setText(self.folder_path)
            # Reset results if a new folder is selected
            self.result_data = {}
            self.image_label.setText("Folder selected. Ready to start.")
            self._update_button_states()

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if file_path:
            self.model_path = file_path
            self.model_path_edit.setText(self.model_path)
            self._update_button_states()

    def start_detection(self):
        if not self.folder_path or not os.path.isdir(self.folder_path):
            QMessageBox.warning(self, "Warning", "Please select a valid image folder first.")
            return
        if not self.model_path or not os.path.isfile(self.model_path):
             QMessageBox.warning(self, "Warning", "Please select a valid YOLO model file (.pt).")
             return

        # Check for existing thread (shouldn't happen with proper state mgmt, but good practice)
        if self.is_processing:
            QMessageBox.information(self, "Info", "Processing is already in progress.")
            return

        # Get list of image files
        try:
            all_files = os.listdir(self.folder_path)
            self.current_files = [
                f for f in all_files
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ]
            if not self.current_files:
                 QMessageBox.warning(self, "Warning", "No supported image files found in the selected folder.")
                 return
        except OSError as e:
             QMessageBox.critical(self, "Error", f"Could not read folder contents: {e}")
             return


        self.is_processing = True
        self.pause_button.setText("Pause") # Reset pause button text
        self.result_data = {} # Clear previous results
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.current_files))
        self._update_button_states()

        # Setup and start the worker thread
        self.worker_thread = QThread()
        self.worker = Worker(self.model_path, self.folder_path, self.current_files)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals from worker to slots in the main GUI thread
        self.worker.progressUpdated.connect(self.update_progress)
        self.worker.statusUpdated.connect(self.update_status)
        self.worker.imageProcessed.connect(self.display_processed_image)
        self.worker.processingFinished.connect(self.on_processing_finished)
        self.worker.errorOccurred.connect(self.show_error)

        # Connect thread signals for lifecycle management
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater) # Schedule thread deletion
        self.worker.processingFinished.connect(self.on_worker_finished)

        self.worker_thread.start()

    def toggle_pause(self):
        if not self.is_processing or not self.worker:
            return

        if self.worker._is_paused:
            self.worker.resume()
            self.pause_button.setText("Pause")
        else:
            self.worker.pause()
            self.pause_button.setText("Resume")

    def stop_detection(self):
        if self.worker:
            self.worker.stop() # Signal the worker to stop
        # State update and cleanup will happen in on_worker_finished

    def draw_and_save_images(self):
        if not self.result_data:
            QMessageBox.warning(self, "No Results", "No detection results available. Run detection first.")
            return
        if not self.folder_path:
             QMessageBox.warning(self, "No Folder", "Original folder path not set.")
             return

        results_dir = os.path.join(self.folder_path, RESULTS_FOLDER_NAME)
        try:
            os.makedirs(results_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not create results directory '{results_dir}': {e}")
            return

        self.status_label.setText(f"Drawing and saving annotated images to {results_dir}...")
        QApplication.processEvents() # Update label immediately

        saved_count = 0
        error_count = 0
        total_items = len(self.result_data.items())
        self.progress_bar.setMaximum(total_items)
        self.progress_bar.setValue(0)

        # Potentially long operation - could also be moved to a thread if needed
        # For now, run in main thread with progress updates
        for i, (filename, detections) in enumerate(self.result_data.items()):
            original_path = os.path.join(self.folder_path, filename)
            result_path = os.path.join(results_dir, filename)

            try:
                image = cv2.imread(original_path)
                if image is None:
                    print(f"Warning: Could not read original image {filename} during save. Skipping.")
                    error_count += 1
                    continue

                # Re-draw detections (could also store annotated images in worker if memory allows)
                for detection in detections:
                    box = detection["box"] # Already int from worker
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    color = (0, 255, 0) # Example color
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    label_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 15
                    cv2.putText(image, label, (box[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Save the annotated image
                success = cv2.imwrite(result_path, image)
                if not success:
                     print(f"Warning: Could not write annotated image {result_path}. Skipping.")
                     error_count += 1
                else:
                    saved_count +=1

            except Exception as e:
                print(f"Error processing/saving {filename} for annotation: {e}")
                error_count += 1

            self.progress_bar.setValue(i + 1)
            # Allow GUI to remain responsive during saving
            if i % 10 == 0: # Process events every 10 images
                 QApplication.processEvents()


        self.status_label.setText("Annotation saving complete.")
        QMessageBox.information(self, "Save Complete",
                                f"Saved {saved_count} annotated images to '{results_dir}'.\n"
                                f"{error_count} errors occurred.")
        self.progress_bar.setValue(0) # Reset progress bar


    # --- Slot Methods (Called by Worker Signals) ---
    def update_progress(self, value, max_value):
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def display_processed_image(self, image_path, annotated_cv_image):
        if annotated_cv_image is None:
             # Maybe display a placeholder or the original image?
             # For now, just indicate it couldn't be processed/displayed
             # pixmap = QPixmap(image_path) # Show original if annotation failed?
             self.image_label.setText(f"Processed (or failed): {os.path.basename(image_path)}\n(No preview)")
             return

        try:
            # Convert OpenCV image (BGR) to QImage (RGB)
            height, width, channel = annotated_cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_cv_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            # Important: Use rgbSwapped() if the source is BGR like OpenCV's default
            q_image = q_image.rgbSwapped()

            pixmap = QPixmap.fromImage(q_image)

            # Scale pixmap to fit the label while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
             print(f"Error displaying image {os.path.basename(image_path)}: {e}")
             self.image_label.setText(f"Error displaying: {os.path.basename(image_path)}")


    def on_processing_finished(self, results_data):
        self.result_data = results_data
        self.is_processing = False
        # self.worker = None # Worker will be cleaned up via finished signal
        # self.worker_thread = None
        self._save_results_to_json()
        self._update_button_states()
        QMessageBox.information(self, "Completed", "Object detection finished successfully!")


    def on_worker_finished(self):
        """Called when the worker's run() method finishes or is stopped"""
        # This might be called even if stopped early
        self.is_processing = False
        if self.worker_thread: # Ensure thread exists before quitting
             self.worker_thread.quit() # Ask the thread's event loop to exit
             self.worker_thread.wait() # Wait for thread to fully terminate
        self.worker = None # Clear references
        self.worker_thread = None
        self._update_button_states()
        # Status label might already be set to "Stopped" or "Finished" by worker
        # self.status_label.setText("Idle.") # Reset status if needed


    def show_error(self, error_message):
        QMessageBox.critical(self, "Processing Error", error_message)
        # Optionally stop processing on critical errors
        # self.stop_detection()


    # --- Utility Methods ---
    def _update_button_states(self):
        """Enable/disable buttons based on current state"""
        folder_selected = bool(self.folder_path)
        model_selected = bool(self.model_path)

        self.start_button.setEnabled(folder_selected and model_selected and not self.is_processing)
        self.pause_button.setEnabled(self.is_processing)
        self.stop_button.setEnabled(self.is_processing)

        # Only enable draw/save if results exist and not currently processing
        self.draw_save_button.setEnabled(bool(self.result_data) and not self.is_processing)


    def _save_results_to_json(self):
        if not self.result_data or not self.folder_path:
            return # Nothing to save or nowhere to save it

        save_path = os.path.join(self.folder_path, JSON_RESULTS_FILENAME)
        try:
            with open(save_path, "w") as f:
                json.dump(self.result_data, f, indent=4)
            print(f"Detection results saved to {save_path}") # Log to console
            self.status_label.setText(f"Results saved to {JSON_RESULTS_FILENAME}")
        except Exception as e:
            self.show_error(f"Error saving results to JSON: {e}")


    def closeEvent(self, event):
        """Ensure worker thread is stopped cleanly when closing the window"""
        if self.is_processing and self.worker:
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "Processing is ongoing. Stop and exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_detection()
                # Wait briefly for thread termination signals if possible
                if self.worker_thread:
                    self.worker_thread.wait(1000) # Wait up to 1 sec
                event.accept() # Close the window
            else:
                event.ignore() # Don't close
        else:
            event.accept() # No processing, close normally


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set a default font if desired
    # font = QFont("Segoe UI", 10)
    # app.setFont(font)

    main_window = ObjectDetectionApp()
    main_window.show()
    sys.exit(app.exec())

# --- END OF FILE object-detect-pyqt6.py ---