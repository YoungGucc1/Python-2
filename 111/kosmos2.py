import sys
import os
import random
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.ImageQt import ImageQt
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2
import pandas as pd
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QComboBox, QSplitter, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QColor, QTextCursor, QIcon

# --- Constants and Helper Functions (from original app.py) ---

# Slightly adjusted colors for better visibility perhaps, or use original
colors = [
    (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (114, 128, 250), (0, 165, 255), (0, 128, 0), (144, 238, 144),
    (238, 238, 175), (255, 191, 0), (0, 128, 0), (226, 43, 138),
    (255, 0, 255), (0, 215, 255), (255, 0, 0),
    (128, 0, 0), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (0, 0, 128), (255, 165, 0), (218, 112, 214), (50, 205, 50)
]

color_map_hex = {
    f"{color_id}": f"#{hex(color[0])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[2])[2:].zfill(2)}"
    for color_id, color in enumerate(colors)
}

# Ensure results directories exist
RESULTS_DIR = "results"
RESULTS_IMG_DIR = os.path.join(RESULTS_DIR, "images")
RESULTS_EXCEL = os.path.join(RESULTS_DIR, "captions.xlsx")
os.makedirs(RESULTS_IMG_DIR, exist_ok=True)


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities):
    """ Draws entity boxes on the image. Returns a PIL Image. """
    if isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Invalid image path: {image}")
    elif isinstance(image, Image.Image):
        pil_img = image.copy()
    else:
        raise ValueError(f"Invalid image type: {type(image)}")

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    image_cv = np.array(pil_img)[:, :, ::-1].copy() # RGB to BGR for OpenCV
    image_h, image_w = image_cv.shape[:2]

    if not entities:
        return pil_img # Return original if no entities

    # Limit entities to the number of available colors
    entities = entities[:len(colors)]

    new_image = image_cv # Work on the OpenCV image
    previous_bboxes = []
    text_size = 0.6 # Adjusted for typical viewing
    text_line = 1
    box_line = 2 # Adjusted thickness

    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    color_id = -1
    for entity_idx, (entity_name, (start, end), bboxes) in enumerate(entities):
        color_id += 1
        if not bboxes: continue # Skip if no bounding boxes for this entity

        # Use the first bounding box for label placement consistency
        x1_norm, y1_norm, x2_norm, y2_norm = bboxes[0]

        orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)

        # Draw all bboxes for the entity
        color = colors[color_id % len(colors)]
        for (xb1, yb1, xb2, yb2) in bboxes:
             cv2.rectangle(new_image, (int(xb1 * image_w), int(yb1 * image_h)), (int(xb2 * image_w), int(yb2 * image_h)), color, box_line)

        # --- Text Labeling (simplified for clarity) ---
        label = f"{entity_name}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)

        # Position label slightly above the top-left corner of the first box
        text_x = orig_x1 + box_line
        text_y = orig_y1 - box_line - baseline

        # Adjust if label goes off-screen (top)
        if text_y < text_height:
            text_y = orig_y1 + text_height + baseline + box_line

        # Simple background rectangle
        bg_y1 = text_y - text_height - baseline
        bg_y2 = text_y + baseline
        bg_x1 = text_x
        bg_x2 = text_x + text_width

        # Clamp background rectangle to image bounds
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(image_h, bg_y2)
        bg_x1 = max(0, bg_x1)
        bg_x2 = min(image_w, bg_x2)

        # Basic overlap avoidance (doesn't guarantee perfect placement)
        for prev_bbox in previous_bboxes:
             if is_overlapping((bg_x1, bg_y1, bg_x2, bg_y2), prev_bbox):
                 # Shift down if overlapping
                 shift = prev_bbox[3] - bg_y1 + 2 # Shift below the previous box + margin
                 bg_y1 += shift
                 bg_y2 += shift
                 text_y += shift
                 if bg_y2 > image_h: # If shifted off bottom, try placing inside the box
                     text_y = orig_y1 + text_height + baseline + box_line
                     bg_y1 = text_y - text_height - baseline
                     bg_y2 = text_y + baseline
                 break # Check against next previous_bboxes with new position


        alpha = 0.6 # Transparency
        overlay = new_image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1) # Filled rectangle
        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        # Put text
        cv2.putText(new_image, label, (text_x, text_y - baseline//2), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA)
        previous_bboxes.append((bg_x1, bg_y1, bg_x2, bg_y2))
        # --- End Text Labeling ---

    # Convert back to PIL Image (BGR to RGB)
    final_pil_image = Image.fromarray(new_image[:, :, ::-1])
    return final_pil_image

def pil_to_qpixmap(pil_image):
    """Converts a PIL Image to a QPixmap."""
    if pil_image.mode == "RGB":
        r, g, b = pil_image.split()
        pil_image = Image.merge("RGB", (b, g, r))
    elif pil_image.mode == "RGBA":
        r, g, b, a = pil_image.split()
        pil_image = Image.merge("RGBA", (b, g, r, a))

    img_qt = ImageQt(pil_image)
    qimage = QImage(img_qt)
    # qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)

# --- Worker Threads ---

class ModelLoader(QObject):
    """Loads the model in a separate thread."""
    model_ready = pyqtSignal(object, object) # processor, model
    error = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            ckpt = "microsoft/kosmos-2-patch14-224"
            model = AutoModelForVision2Seq.from_pretrained(ckpt).to(device)
            processor = AutoProcessor.from_pretrained(ckpt)
            print("Model loaded successfully.")
            self.model_ready.emit(processor, model)
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")

class ProcessWorker(QObject):
    """Processes images in a separate thread."""
    progress = pyqtSignal(int) # current item index
    finished_item = pyqtSignal(str, object, str, list) # orig_path, annotated_img (PIL), processed_text, entities
    finished_batch = pyqtSignal(list) # List of results: (orig_path, annotated_path, caption)
    error = pyqtSignal(str)

    def __init__(self, processor, model, image_paths, text_prompt_mode, device):
        super().__init__()
        self.processor = processor
        self.model = model
        self.image_paths = image_paths
        self.text_prompt_mode = text_prompt_mode
        self.device = device
        self._is_running = True

    def stop(self):
        self._is_running = False

    @pyqtSlot()
    def run(self):
        results_data = []
        total_items = len(self.image_paths)

        if self.text_prompt_mode == "Brief":
            base_text_input = "<grounding>An image of"
        elif self.text_prompt_mode == "Detailed":
            base_text_input = "<grounding>Describe this image in detail:"
        else: # Should not happen with ComboBox but handle anyway
             base_text_input = "<grounding>"

        for idx, img_path in enumerate(self.image_paths):
            if not self._is_running:
                self.error.emit("Processing cancelled.")
                break
            try:
                self.progress.emit(idx)
                # Load image
                image_input = Image.open(img_path).convert("RGB")

                # Prepare inputs
                inputs = self.processor(text=base_text_input, images=image_input, return_tensors="pt").to(self.device)

                # Generate
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=128,
                )

                # Post-process
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                processed_text, entities = self.processor.post_process_generation(generated_text)

                # Filter entities (only those with text spans)
                filtered_entities = [e for e in entities if e[1][0] is not None and e[1][0] != e[1][1]]

                # Draw boxes
                annotated_image_pil = draw_entity_boxes_on_image(image_input, filtered_entities)

                # Save annotated image
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                annotated_filename = f"{base_filename}_annotated.jpg"
                annotated_path = os.path.join(RESULTS_IMG_DIR, annotated_filename)
                annotated_image_pil.save(annotated_path, "JPEG")

                # Store result for Excel
                results_data.append({
                    "Original Image Path": img_path,
                    "Annotated Image Path": annotated_path,
                    "Generated Caption": processed_text
                })

                # Emit result for UI update
                self.finished_item.emit(img_path, annotated_image_pil, processed_text, filtered_entities)

                # Give GUI a chance to update
                QApplication.processEvents()
                time.sleep(0.05) # Small delay

            except FileNotFoundError:
                self.error.emit(f"Error: Image file not found: {img_path}")
            except Exception as e:
                self.error.emit(f"Error processing {os.path.basename(img_path)}: {e}")
                # Continue with the next image if possible

        if self._is_running:
            self.progress.emit(total_items) # Signal completion
            self.finished_batch.emit(results_data)
        else:
            # Clean up partial results if cancelled? Or save what was done?
            # For now, just signal cancellation error.
             pass


# --- Main Application Window ---

class Kosmos2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kosmos-2 Batch Processor")
        self.setGeometry(100, 100, 1200, 700)

        # Model and Processor placeholders
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._current_worker = None
        self._worker_thread = None
        self._model_loader_thread = None

        self._selected_entities = [] # Store entities for the currently selected image
        self._current_image_path = None # Path of the image shown in detail view

        # --- UI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel (Controls & File List)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_panel.setFixedWidth(350)

        self.btn_select_images = QPushButton(QIcon.fromTheme("document-open"), "Select Image(s)")
        self.btn_select_folder = QPushButton(QIcon.fromTheme("folder-open"), "Select Folder")
        self.btn_clear_list = QPushButton(QIcon.fromTheme("edit-clear"), "Clear List")

        self.file_list_widget = QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(self.on_file_selected)

        self.prompt_label = QLabel("Description Type:")
        self.prompt_combo = QComboBox()
        self.prompt_combo.addItems(["Brief", "Detailed"])
        self.prompt_combo.setCurrentText("Brief")

        self.btn_run = QPushButton(QIcon.fromTheme("system-run"), "Run Processing")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_run.setEnabled(False) # Disabled until model is loaded

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)

        # Add widgets to left layout
        self.left_layout.addWidget(self.btn_select_images)
        self.left_layout.addWidget(self.btn_select_folder)
        self.left_layout.addWidget(self.btn_clear_list)
        self.left_layout.addWidget(self.file_list_widget)
        self.left_layout.addWidget(self.prompt_label)
        self.left_layout.addWidget(self.prompt_combo)
        self.left_layout.addSpacing(20)
        self.left_layout.addWidget(self.btn_run)
        self.left_layout.addWidget(self.progress_bar)

        # Right Panel (Image Display & Caption)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Image Viewer Area
        self.image_viewer_area = QScrollArea()
        self.image_viewer_area.setWidgetResizable(True)
        self.image_label = QLabel("Select an image from the list or run processing.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300) # Min size for image display
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow scaling
        self.image_viewer_area.setWidget(self.image_label)


        # Bottom: Caption Area
        self.caption_area = QWidget()
        self.caption_layout = QVBoxLayout(self.caption_area)
        self.caption_label = QLabel("Generated Caption:")
        self.caption_textedit = QTextEdit()
        self.caption_textedit.setReadOnly(True)
        self.caption_layout.addWidget(self.caption_label)
        self.caption_layout.addWidget(self.caption_textedit)


        self.splitter.addWidget(self.image_viewer_area)
        self.splitter.addWidget(self.caption_area)
        self.splitter.setSizes([450, 250]) # Initial sizes for splitter panes

        self.right_layout.addWidget(self.splitter)

        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel, 1) # Allow right panel to stretch

        # Status Bar
        self.statusBar().showMessage("Ready. Loading Model...")

        # --- Connections ---
        self.btn_select_images.clicked.connect(self.select_images)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_clear_list.clicked.connect(self.clear_file_list)
        self.btn_run.clicked.connect(self.start_processing)

        # --- Load Model ---
        self.load_model_async()

    # --- UI Slots ---

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if files:
            current_items = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
            added_count = 0
            for file in files:
                if file not in current_items:
                    self.file_list_widget.addItem(file)
                    added_count += 1
            if added_count > 0:
                self.statusBar().showMessage(f"Added {added_count} image(s) to the list.")
            if self.file_list_widget.count() > 0:
                self.file_list_widget.setCurrentRow(0) # Select first item


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            current_items = {self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())}
            added_count = 0
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    file_path = os.path.join(folder, filename)
                    if file_path not in current_items:
                        self.file_list_widget.addItem(file_path)
                        added_count += 1
            if added_count > 0:
                 self.statusBar().showMessage(f"Added {added_count} image(s) from folder.")
            if self.file_list_widget.count() > 0:
                self.file_list_widget.setCurrentRow(0)

    def clear_file_list(self):
        self.file_list_widget.clear()
        self.image_label.setText("Image list cleared.")
        self.image_label.setPixmap(QPixmap()) # Clear pixmap
        self.caption_textedit.clear()
        self.statusBar().showMessage("File list cleared.")
        self._selected_entities = []
        self._current_image_path = None


    def on_file_selected(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        file_path = item.text()
        self._current_image_path = file_path

        # Attempt to load and display the original image immediately
        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                 self.image_label.setText(f"Cannot display:\n{os.path.basename(file_path)}")
                 self.image_label.setPixmap(QPixmap())
            else:
                 scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                 self.image_label.setPixmap(scaled_pixmap)
                 self.image_label.setText("") # Clear any previous text

            # Clear caption until processing is done for this image
            self.caption_textedit.clear()
            self._selected_entities = []

        except Exception as e:
            self.image_label.setText(f"Error loading preview:\n{e}")
            self.image_label.setPixmap(QPixmap())
            self.caption_textedit.clear()
            self._selected_entities = []


    def resizeEvent(self, event):
        """Handle window resize to rescale the displayed image."""
        if self._current_image_path and not self.image_label.pixmap().isNull():
            # Reload the original pixmap and scale it to the new label size
            try:
                original_pixmap = QPixmap(self._current_image_path)
                if not original_pixmap.isNull():
                    scaled_pixmap = original_pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
            except Exception:
                pass # Ignore if reloading fails during resize
        super().resizeEvent(event)


    # --- Model Loading ---

    def load_model_async(self):
        self._model_loader_thread = QThread()
        loader = ModelLoader()
        loader.moveToThread(self._model_loader_thread)

        loader.model_ready.connect(self.on_model_loaded)
        loader.error.connect(self.on_model_load_error)
        self._model_loader_thread.started.connect(loader.run)
        self._model_loader_thread.finished.connect(self._model_loader_thread.deleteLater) # Clean up thread

        self._model_loader_thread.start()

    @pyqtSlot(object, object)
    def on_model_loaded(self, processor, model):
        self.processor = processor
        self.model = model
        self.statusBar().showMessage(f"Model loaded successfully on {self.device}. Ready to process.")
        self.btn_run.setEnabled(True)
        if self._model_loader_thread:
             self._model_loader_thread.quit() # Stop the thread event loop


    @pyqtSlot(str)
    def on_model_load_error(self, error_message):
        self.statusBar().showMessage(f"Model loading failed: {error_message}")
        self.btn_run.setEnabled(False)
        # Optionally show a dialog box
        print(f"ERROR: {error_message}")
        if self._model_loader_thread:
             self._model_loader_thread.quit()


    # --- Processing Logic ---

    def start_processing(self):
        if not self.processor or not self.model:
            self.statusBar().showMessage("Error: Model not loaded.")
            return

        image_paths = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
        if not image_paths:
            self.statusBar().showMessage("No images selected for processing.")
            return

        # Stop existing worker if any
        self.stop_processing() # Ensure any previous run is stopped

        # Disable UI during processing
        self.set_ui_enabled(False)
        self.statusBar().showMessage("Starting processing...")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(image_paths))

        # Create and start worker thread
        self._worker_thread = QThread()
        self._current_worker = ProcessWorker(
            self.processor, self.model, image_paths, self.prompt_combo.currentText(), self.device
        )
        self._current_worker.moveToThread(self._worker_thread)

        # Connect signals
        self._current_worker.progress.connect(self.update_progress)
        self._current_worker.finished_item.connect(self.display_processed_item)
        self._current_worker.finished_batch.connect(self.on_batch_finished)
        self._current_worker.error.connect(self.on_processing_error)
        self._worker_thread.started.connect(self._current_worker.run)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater) # Clean up thread

        self._worker_thread.start()

    def stop_processing(self):
         if self._current_worker:
             self._current_worker.stop()
             # Note: The worker thread will exit gracefully when it checks _is_running
             # We don't forcibly terminate it.
         if self._worker_thread and self._worker_thread.isRunning():
             print("Requesting worker termination...")
             # Thread will finish its current loop check and exit run()
             # It will emit signals upon completion/error if needed.


    def set_ui_enabled(self, enabled):
        """Enable/disable UI elements during processing."""
        self.btn_select_images.setEnabled(enabled)
        self.btn_select_folder.setEnabled(enabled)
        self.btn_clear_list.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)
        self.prompt_combo.setEnabled(enabled)
        self.btn_run.setEnabled(enabled) # Re-enable run button when done/error

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        total = self.progress_bar.maximum()
        if value < total:
            self.statusBar().showMessage(f"Processing image {value + 1} of {total}...")
        else:
             self.statusBar().showMessage("Processing finished.")


    @pyqtSlot(str, object, str, list)
    def display_processed_item(self, orig_path, annotated_img_pil, processed_text, entities):
        """Update UI with the result of a single processed image."""
        # Find the list item corresponding to this path
        items = self.file_list_widget.findItems(orig_path, Qt.MatchFlag.MatchExactly)
        if items:
            item = items[0]
            item.setData(Qt.ItemDataRole.UserRole + 1, annotated_img_pil) # Store PIL image
            item.setData(Qt.ItemDataRole.UserRole + 2, processed_text)
            item.setData(Qt.ItemDataRole.UserRole + 3, entities)
            item.setForeground(QColor("green")) # Mark as processed

            # If this is the currently selected item, update the display
            if self.file_list_widget.currentItem() == item:
                qpixmap = pil_to_qpixmap(annotated_img_pil)
                scaled_pixmap = qpixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.display_highlighted_text(processed_text, entities)
                self._selected_entities = entities # Update entities for current view
                self._current_image_path = orig_path # Update path in case it was null

    def display_highlighted_text(self, processed_text, entities):
        """ Creates HTML for highlighted text and sets it in the QTextEdit """
        if not processed_text:
            self.caption_textedit.setPlainText("")
            return

        colored_text_segments = []
        entity_info = []
        color_id_counter = -1

        # Sort entities by start position to process text linearly
        sorted_entities = sorted([e for e in entities if e[1][0] is not None], key=lambda x: x[1][0])

        for entity in sorted_entities:
            entity_name, (start, end), bboxes = entity
            if start == end: # Skip entities without a phrase
                 continue
            color_id_counter += 1
            entity_info.append(((start, end), color_id_counter % len(colors))) # Use modulo for color cycling

        # Build HTML string
        html_output = ""
        prev_end = 0
        for (start, end), color_id in entity_info:
            # Add text before the current entity (if any)
            if start > prev_end:
                html_output += processed_text[prev_end:start].replace("<", "<").replace(">", ">")

            # Add the highlighted entity text
            entity_text = processed_text[start:end].replace("<", "<").replace(">", ">")
            # Use the hex color map directly
            hex_color = color_map_hex.get(str(color_id % len(color_map_hex)), "#000000") # Fallback to black
            # Simple background highlight
            html_output += f'<span style="background-color: {hex_color}; color: black; padding: 1px 0px; border-radius: 2px;">{entity_text}</span>'
            prev_end = end

        # Add any remaining text after the last entity
        if prev_end < len(processed_text):
            html_output += processed_text[prev_end:].replace("<", "<").replace(">", ">")

        self.caption_textedit.setHtml(f"<pre>{html_output}</pre>") # Use <pre> for monospace/consistent spacing


    @pyqtSlot(list)
    def on_batch_finished(self, results_data):
        """Called when the entire batch is processed."""
        self.statusBar().showMessage(f"Batch processing complete. Saved {len(results_data)} results.")
        self.set_ui_enabled(True)

        # Save results to Excel
        if results_data:
            try:
                df = pd.DataFrame(results_data)
                # Append if file exists, otherwise create new
                if os.path.exists(RESULTS_EXCEL):
                     try:
                         existing_df = pd.read_excel(RESULTS_EXCEL)
                         df = pd.concat([existing_df, df], ignore_index=True)
                     except Exception as read_err:
                         print(f"Warning: Could not read existing Excel file {RESULTS_EXCEL}. Overwriting. Error: {read_err}")
                         # Fall through to overwrite if reading fails

                df.to_excel(RESULTS_EXCEL, index=False, engine='openpyxl')
                self.statusBar().showMessage(f"Batch finished. Results saved to {RESULTS_DIR}/ and {RESULTS_EXCEL}")
            except Exception as e:
                self.statusBar().showMessage(f"Batch finished, but failed to save Excel: {e}")
                print(f"ERROR saving Excel: {e}")
        else:
             self.statusBar().showMessage("Batch finished. No results were generated.")


        # Clean up worker references
        if self._worker_thread:
             self._worker_thread.quit()
        self._worker_thread = None
        self._current_worker = None


    @pyqtSlot(str)
    def on_processing_error(self, error_message):
        """Handles errors reported by the worker thread."""
        self.statusBar().showMessage(f"Error: {error_message}")
        print(f"ERROR: {error_message}")
        self.set_ui_enabled(True) # Re-enable UI on error
        self.progress_bar.setValue(0) # Reset progress

        # Clean up worker references
        if self._worker_thread:
             self._worker_thread.quit()
        self._worker_thread = None
        self._current_worker = None


    def closeEvent(self, event):
        """Ensure threads are stopped on closing."""
        self.stop_processing()
        if self._model_loader_thread and self._model_loader_thread.isRunning():
             self._model_loader_thread.quit()
             self._model_loader_thread.wait(1000) # Wait a bit for it to finish
        super().closeEvent(event)

# --- Main Execution ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # You can set a style like Fusion for a more modern look across platforms
    # app.setStyle("Fusion")
    window = Kosmos2Window()
    window.show()
    sys.exit(app.exec())