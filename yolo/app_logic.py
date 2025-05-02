"""
AppLogic module - Connects UI events to backend logic (PyQt6 version)
"""

import os
import cv2
from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QMutex, QMutexLocker, QRect
from PyQt6.QtWidgets import QMessageBox # Use directly for icons
import numpy as np
from typing import List, Dict, Any, Optional

# Use local imports due to flat structure
from data_handler import DataHandler
from yolo_processor import YoloProcessor
from image_augmenter import ImageAugmenter
from main_window import MainWindow # Need main window type hint

# --- Worker Threads ---

class WorkerSignals(QObject):
    """Defines signals available from worker threads."""
    progress = pyqtSignal(int, int, str)  # current, total, status_message
    result = pyqtSignal(object)           # Emit results (type depends on worker)
    error = pyqtSignal(str)               # Emit error message
    finished = pyqtSignal()               # Emit when finished

class DetectionWorker(QThread):
    """Worker thread for running object detection."""
    def __init__(self, yolo_processor: YoloProcessor, image_path: str, config: dict):
        super().__init__()
        self.signals = WorkerSignals()
        self.yolo_processor = yolo_processor
        self.image_path = image_path
        self.conf_threshold = config.get('conf_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self._is_running = True

    def run(self):
        try:
            if not self._is_running: return
            self.signals.progress.emit(0, 1, f"Detecting: {os.path.basename(self.image_path)}...")

            detections = self.yolo_processor.detect(
                self.image_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )

            if not self._is_running: return # Check again after potentially long detection

            if detections is None:
                self.signals.error.emit(f"Detection failed for {os.path.basename(self.image_path)}")
            else:
                self.signals.progress.emit(1, 1, f"Detected {len(detections)} objects")
                self.signals.result.emit({'image_path': self.image_path, 'detections': detections})

        except Exception as e:
            self.signals.error.emit(f"Detection Error: {e}")
        finally:
            self.signals.finished.emit()

    def stop(self):
        self._is_running = False


class AugmentationWorker(QThread):
    """Worker thread for augmenting images."""
    def __init__(self, image_augmenter: ImageAugmenter, data_handler: DataHandler, num_augmentations: int):
        super().__init__()
        self.signals = WorkerSignals()
        self.image_augmenter = image_augmenter
        self.data_handler = data_handler # Access data handler for image paths/annotations
        self.num_augmentations = num_augmentations
        self._is_running = True
        # No mutex needed here if results are collected and emitted at the end

    def run(self):
        all_augmented_results = {} # Collect results here: {image_path: [(img_np, anns_list), ...]}
        try:
            # Get images that have annotations with assigned classes
            annotated_image_paths = [
                path for path, annotations in self.data_handler.annotations.items()
                if any(ann.get('class_id', -1) >= 0 for ann in annotations)
            ]

            if not annotated_image_paths:
                self.signals.error.emit("No images with assigned annotations to augment.")
                return

            total_images = len(annotated_image_paths)
            self.signals.progress.emit(0, total_images, f"Starting augmentation for {total_images} images...")

            for i, image_path in enumerate(annotated_image_paths):
                if not self._is_running: return
                self.signals.progress.emit(i, total_images, f"Augmenting {i+1}/{total_images}: {os.path.basename(image_path)}")

                annotations = self.data_handler.annotations[image_path]
                img_bgr = cv2.imread(image_path)

                if img_bgr is None or not annotations:
                    print(f"Warning: Skipping augmentation for {image_path} (load fail or no annotations).")
                    continue

                augmented_data = self.image_augmenter.augment_image(
                    img_bgr, annotations, self.num_augmentations
                )

                if augmented_data:
                    all_augmented_results[image_path] = augmented_data

                # Optional: Yield some CPU time if augmentation is very intensive per image
                # self.msleep(10)

            if not self._is_running: return # Check before final emit

            self.signals.progress.emit(total_images, total_images, "Augmentation finished.")
            self.signals.result.emit(all_augmented_results)

        except Exception as e:
            self.signals.error.emit(f"Augmentation Error: {e}")
        finally:
            self.signals.finished.emit()

    def stop(self):
        self._is_running = False


class SaveDatasetWorker(QThread):
    """Worker thread for saving the dataset."""
    def __init__(self, data_handler: DataHandler, output_dir: str, config: dict):
        super().__init__()
        self.signals = WorkerSignals()
        self.data_handler = data_handler
        self.output_dir = output_dir
        self.train_split = config.get('train_split', 0.8)
        self._is_running = True

    def run(self):
        try:
            def progress_callback(current, total, message):
                if not self._is_running: raise InterruptedError("Save operation cancelled")
                self.signals.progress.emit(current, total, message)

            success = self.data_handler.save_yolo_dataset(
                self.output_dir,
                train_split=self.train_split,
                progress_callback=progress_callback
            )

            if not self._is_running: return

            if success:
                self.signals.result.emit(os.path.join(self.output_dir, 'yolo_dataset')) # Return actual dataset path
            else:
                # Error should have been reported via callback or caught exception
                 if not self.signals._error_emitted: # Check if error already sent
                      self.signals.error.emit("Failed to save dataset (unknown reason).")

        except InterruptedError:
             self.signals.error.emit("Dataset save cancelled.")
        except Exception as e:
            self.signals.error.emit(f"Save Dataset Error: {e}")
        finally:
            self.signals.finished.emit()

    def stop(self):
        self._is_running = True # Let the progress callback handle interruption
        self.signals._error_emitted = False # Helper flag

# --- App Logic Controller ---

class AppLogic(QObject):
    """Application logic controller connecting UI and backend."""

    # Signals to update UI (more specific than before)
    statusChanged = pyqtSignal(str, int) # message, timeout (0=persistent)
    progressChanged = pyqtSignal(int, int) # current, total
    imageDisplayRequested = pyqtSignal(str) # image_path
    annotationsChanged = pyqtSignal(list) # New annotations for current image
    classListChanged = pyqtSignal(list) # Full list of class names
    uiStateUpdated = pyqtSignal(dict) # Send dict of enabled states for buttons etc.
    configUpdateRequest = pyqtSignal(dict) # Request UI update config values (e.g., after load)

    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window
        self.data_handler = DataHandler()
        self.yolo_processor = YoloProcessor()
        self.image_augmenter = ImageAugmenter()

        # State
        self.current_image_path: Optional[str] = None
        self.current_config: Dict = self.main_window.get_config() # Initial config
        self._is_processing: bool = False # Flag for background tasks
        self._current_worker: Optional[QThread] = None

        self._connect_signals()
        self._update_ui_state() # Initial UI state

    def _connect_signals(self):
        """Connect signals from UI and workers to logic slots."""
        # UI -> Logic
        mw = self.main_window
        mw.addImagesClicked.connect(self._on_add_images)
        mw.selectModelClicked.connect(self._on_select_model)
        mw.processImageClicked.connect(self._on_process_image)
        mw.saveDatasetClicked.connect(self._on_save_dataset)
        mw.augmentDatasetClicked.connect(self._on_augment_dataset)
        mw.addClassClicked.connect(self._on_add_class)
        mw.removeClassClicked.connect(self._on_remove_class)
        mw.imageSelected.connect(self._on_image_selected_in_list)
        mw.classSelected.connect(self._on_class_selected_in_list)
        mw.configChanged.connect(self._on_config_changed)
        mw.drawBoxClicked.connect(self._on_draw_box_toggled)
        mw.deleteSelectedBoxClicked.connect(self._on_delete_selected_box)
        mw.image_canvas.newBoxDrawn.connect(self._on_new_box_drawn)
        mw.image_canvas.boxSelected.connect(self._on_box_selected_in_canvas) # Handle direct canvas selection


        # Logic -> UI (via signals defined in AppLogic)
        self.statusChanged.connect(mw.set_status)
        self.progressChanged.connect(mw.set_progress)
        self.imageDisplayRequested.connect(mw.display_image)
        self.annotationsChanged.connect(mw.update_annotations)
        self.classListChanged.connect(mw.update_class_names_display) # Use specific method
        # self.uiStateUpdated.connect(mw._update_buttons_state) # Let UI manage its own state updates for now


    def _start_worker(self, worker_class, *args):
        """Helper to start a worker thread and manage state."""
        if self._is_processing:
            self.statusChanged.emit("Busy: Another process is running.", 3000)
            return

        self._is_processing = True
        self._update_ui_state(processing=True) # Disable controls

        self._current_worker = worker_class(*args)
        # Connect worker signals dynamically
        self._current_worker.signals.progress.connect(self._handle_worker_progress)
        self._current_worker.signals.result.connect(self._handle_worker_result)
        self._current_worker.signals.error.connect(self._handle_worker_error)
        self._current_worker.signals.finished.connect(self._handle_worker_finished)

        self._current_worker.start()

    # --- UI Action Slots ---

    @pyqtSlot(list)
    def _on_add_images(self, file_paths: List[str]):
        added_count = self.data_handler.add_images(file_paths)
        if added_count > 0:
            self.main_window.add_images_to_list(file_paths) # Let UI handle duplicates visually
            self.statusChanged.emit(f"Added {added_count} valid images.", 3000)
        else:
            self.statusChanged.emit("No new valid images added.", 3000)
        self._update_ui_state()

    @pyqtSlot(str)
    def _on_select_model(self, model_path: str):
        if self.yolo_processor.load_model(model_path):
             if self.data_handler.set_model_path(model_path):
                 self.main_window.set_model_path(model_path)
                 # If model has class names, potentially offer to load them
                 model_classes = self.yolo_processor.class_names
                 if model_classes:
                      reply = QMessageBox.question(self.main_window, "Model Classes Found",
                                                   f"Model contains {len(model_classes)} classes: {', '.join(model_classes[:5])}...\n"
                                                   "Do you want to replace current classes with these?",
                                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                   QMessageBox.StandardButton.No)
                      if reply == QMessageBox.StandardButton.Yes:
                           self.data_handler.class_names = [] # Clear existing
                           self.data_handler.annotations = {} # Clear annotations as class IDs change
                           self.data_handler.augmented_images = {}
                           for name in model_classes: self.data_handler.add_class(name)
                           self.classListChanged.emit(self.data_handler.class_names)
                           # Reload current image if any to clear old boxes
                           if self.current_image_path:
                                self._on_image_selected_in_list(self.current_image_path)
                           self.statusChanged.emit(f"Loaded model and updated classes ({len(model_classes)}).", 5000)
                      else:
                           self.statusChanged.emit(f"Loaded model: {os.path.basename(model_path)}", 5000)

                 else:
                      self.statusChanged.emit(f"Loaded model: {os.path.basename(model_path)}", 5000)
             else: # Should not happen if load_model succeeded
                  self.main_window.show_message("Model Error", "Internal error setting model path.", "error")
        else:
            self.data_handler.set_model_path(None) # Clear if load failed
            self.main_window.set_model_path("")
            self.main_window.show_message("Model Load Error", f"Failed to load model:\n{model_path}", "error")
        self._update_ui_state()


    @pyqtSlot()
    def _on_process_image(self):
        if self.current_image_path and self.yolo_processor.model:
            self._start_worker(DetectionWorker, self.yolo_processor, self.current_image_path, self.current_config)
        elif not self.current_image_path:
             self.statusChanged.emit("No image selected to process.", 3000)
        else:
             self.statusChanged.emit("No model loaded.", 3000)


    @pyqtSlot(str)
    def _on_save_dataset(self, output_dir: str):
         if not self.data_handler.class_names:
              self.main_window.show_message("Save Error", "No classes defined. Cannot save dataset.", "warning")
              return
         # Check if any annotations exist with assigned classes
         has_assigned = any(
             any(a.get('class_id', -1) >= 0 for a in anns)
             for anns in self.data_handler.annotations.values()
         )
         has_assigned_aug = any(
             any(any(a.get('class_id', -1) >= 0 for a in aug_anns) for _, aug_anns in aug_list)
             for aug_list in self.data_handler.augmented_images.values()
         )
         if not has_assigned and not has_assigned_aug:
              self.main_window.show_message("Save Error", "No annotations with assigned classes found. Cannot save dataset.", "warning")
              return

         self._start_worker(SaveDatasetWorker, self.data_handler, output_dir, self.current_config)

    @pyqtSlot(int)
    def _on_augment_dataset(self, num_augmentations: int):
         # Check if any annotations exist with assigned classes
         has_assigned = any(
             any(a.get('class_id', -1) >= 0 for a in anns)
             for anns in self.data_handler.annotations.values()
         )
         if not has_assigned:
              self.main_window.show_message("Augment Error", "No images with assigned classes found to augment.", "warning")
              return

         # Clear previous augmentations before starting new ones
         self.data_handler.clear_augmented_images()

         self._start_worker(AugmentationWorker, self.image_augmenter, self.data_handler, num_augmentations)


    @pyqtSlot(str)
    def _on_add_class(self, class_name: str):
        if self.data_handler.add_class(class_name):
            self.classListChanged.emit(self.data_handler.class_names)
            self.statusChanged.emit(f"Added class: {class_name}", 3000)
        else:
            self.statusChanged.emit(f"Class '{class_name}' already exists.", 3000)
        self._update_ui_state()


    @pyqtSlot(str)
    def _on_remove_class(self, class_name: str):
        if self.data_handler.remove_class(class_name):
            self.classListChanged.emit(self.data_handler.class_names)
            self.statusChanged.emit(f"Removed class: {class_name}", 3000)
            # Refresh current image annotations if displayed
            if self.current_image_path:
                self.annotationsChanged.emit(self.data_handler.annotations.get(self.current_image_path, []))
        else:
            self.statusChanged.emit(f"Failed to remove class: {class_name}", 3000)
        self._update_ui_state()


    @pyqtSlot(str)
    def _on_image_selected_in_list(self, image_path: str):
        if image_path == self.current_image_path: return # No change

        self.current_image_path = image_path
        if image_path:
             self.imageDisplayRequested.emit(image_path)
             self.annotationsChanged.emit(self.data_handler.annotations.get(image_path, []))
             self.statusChanged.emit(f"Selected: {os.path.basename(image_path)}", 0)
             # Reset box selection in canvas when image changes
             self.main_window.image_canvas.select_box(-1)
        else:
             # Clear display if selection is cleared
             self.imageDisplayRequested.emit(None)
             self.annotationsChanged.emit([])
             self.statusChanged.emit("No image selected", 0)
        self._update_ui_state()


    @pyqtSlot(str)
    def _on_class_selected_in_list(self, class_name: str):
        # If a box is already selected in the canvas, assign this class
        selected_box_idx = self.main_window.image_canvas.selected_box_idx
        if selected_box_idx != -1 and class_name and self.current_image_path:
            try:
                class_id = self.data_handler.class_names.index(class_name)
                if self.data_handler.assign_class_to_box(self.current_image_path, selected_box_idx, class_id):
                    self.annotationsChanged.emit(self.data_handler.annotations[self.current_image_path])
                    self.statusChanged.emit(f"Assigned '{class_name}' to selected box.", 3000)
                else:
                    self.statusChanged.emit(f"Failed to assign class.", 3000)
            except ValueError:
                self.statusChanged.emit(f"Error: Class '{class_name}' not found internally.", 3000)
        elif class_name:
             self.statusChanged.emit(f"Class selected: {class_name}", 0)
        else:
             self.statusChanged.emit(f"Class selection cleared", 0)
        self._update_ui_state()


    @pyqtSlot(int)
    def _on_box_selected_in_canvas(self, box_index: int):
         # If a class is already selected in the list, assign it
         selected_class_name = self.main_window.get_selected_class_name()
         if box_index != -1 and selected_class_name and self.current_image_path:
            try:
                class_id = self.data_handler.class_names.index(selected_class_name)
                if self.data_handler.assign_class_to_box(self.current_image_path, box_index, class_id):
                    self.annotationsChanged.emit(self.data_handler.annotations[self.current_image_path])
                    self.statusChanged.emit(f"Assigned '{selected_class_name}' to selected box.", 3000)
                else:
                    self.statusChanged.emit(f"Failed to assign class.", 3000)
            except ValueError:
                 self.statusChanged.emit(f"Error: Class '{selected_class_name}' not found internally.", 3000)
         elif box_index != -1:
              self.statusChanged.emit(f"Box {box_index + 1} selected.", 0)
         else:
              self.statusChanged.emit(f"Box selection cleared.", 0)
         self._update_ui_state()


    @pyqtSlot(dict)
    def _on_config_changed(self, config: dict):
        self.current_config = config
        # Maybe add validation here if needed
        self.statusChanged.emit("Configuration updated.", 1000)


    @pyqtSlot(bool)
    def _on_draw_box_toggled(self, enabled: bool):
        if enabled:
            self.statusChanged.emit("Draw mode enabled. Click and drag on image.", 0)
        else:
            self.statusChanged.emit("Draw mode disabled.", 0)
        self._update_ui_state()


    @pyqtSlot(QRect)
    def _on_new_box_drawn(self, box_rect_img: QRect):
         if self.current_image_path:
              added_index = self.data_handler.add_manual_box(self.current_image_path, box_rect_img)
              if added_index != -1:
                   self.annotationsChanged.emit(self.data_handler.annotations[self.current_image_path])
                   # Automatically select the newly drawn box
                   self.main_window.image_canvas.select_box(added_index)
                   self.statusChanged.emit(f"Added new box [{added_index + 1}]. Select a class to assign.", 3000)
              else:
                   self.statusChanged.emit("Failed to add manual box.", 3000)
         self._update_ui_state()


    @pyqtSlot()
    def _on_delete_selected_box(self):
        box_index = self.main_window.image_canvas.selected_box_idx
        if self.current_image_path and box_index != -1:
            if self.data_handler.delete_box(self.current_image_path, box_index):
                self.main_window.image_canvas.select_box(-1) # Deselect after delete
                self.annotationsChanged.emit(self.data_handler.annotations[self.current_image_path])
                self.statusChanged.emit(f"Deleted box {box_index + 1}.", 3000)
            else:
                self.statusChanged.emit(f"Failed to delete box {box_index + 1}.", 3000)
        else:
             self.statusChanged.emit("No box selected to delete.", 3000)
        self._update_ui_state()

    # --- Worker Signal Handlers ---

    @pyqtSlot(int, int, str)
    def _handle_worker_progress(self, current: int, total: int, message: str):
        self.progressChanged.emit(current, total)
        self.statusChanged.emit(message, 0) # Persistent status during progress

    @pyqtSlot(object)
    def _handle_worker_result(self, result):
        # Determine result type based on worker (could use isinstance or worker type)
        if isinstance(self._current_worker, DetectionWorker) and isinstance(result, dict):
            image_path = result.get('image_path')
            detections = result.get('detections')
            if image_path == self.current_image_path: # Update if still viewing the same image
                self.data_handler.update_annotations_from_detections(image_path, detections)
                self.annotationsChanged.emit(self.data_handler.annotations[image_path])
                self.statusChanged.emit(f"Detection complete for {os.path.basename(image_path)}. Found {len(detections)} objects.", 5000)
            else:
                 # Update data handler even if not currently viewed
                 self.data_handler.update_annotations_from_detections(image_path, detections)
                 self.statusChanged.emit(f"Background detection complete for {os.path.basename(image_path)}.", 3000)

        elif isinstance(self._current_worker, AugmentationWorker) and isinstance(result, dict):
            total_augmented = 0
            for image_path, augmented_data in result.items():
                self.data_handler.add_augmented_images(image_path, augmented_data)
                total_augmented += len(augmented_data)
            self.statusChanged.emit(f"Augmentation complete. Added {total_augmented} variations.", 5000)
            self.main_window.show_message("Augmentation Complete", f"Successfully created {total_augmented} augmented images.")

        elif isinstance(self._current_worker, SaveDatasetWorker) and isinstance(result, str):
             dataset_path = result
             self.statusChanged.emit(f"Dataset saved successfully to: {dataset_path}", 5000)
             self.main_window.show_message("Save Complete", f"Dataset saved to:\n{dataset_path}")

        self._update_ui_state()


    @pyqtSlot(str)
    def _handle_worker_error(self, error_msg: str):
        print(f"Worker Error: {error_msg}")
        self.statusChanged.emit(f"Error: {error_msg}", 5000)
        self.main_window.show_message("Operation Error", error_msg, "error")
        self._update_ui_state() # Re-enable UI potentially


    @pyqtSlot()
    def _handle_worker_finished(self):
        self.progressChanged.emit(0, 100) # Reset progress bar
        self._is_processing = False
        self._current_worker = None
        self._update_ui_state(processing=False) # Re-enable controls

    # --- State Management ---

    def _update_ui_state(self, processing: Optional[bool] = None):
        """Central method to update UI element enabled states."""
        if processing is None:
            processing = self._is_processing

        self.main_window.enable_processing_controls(not processing)
        self.main_window._update_buttons_state() # Trigger main window's detailed update

        # Explicitly enable/disable Augment/Save based on data state
        has_assigned = any(
             any(a.get('class_id', -1) >= 0 for a in anns)
             for anns in self.data_handler.annotations.values()
         )
        # Augmentations only make sense if original images have assigned classes
        self.main_window.augment_btn.setEnabled(has_assigned and not processing)

        # Save makes sense if originals OR augmentations have assigned classes
        has_assigned_aug = any(
             any(any(a.get('class_id', -1) >= 0 for a in aug_anns) for _, aug_anns in aug_list)
             for aug_list in self.data_handler.augmented_images.values()
        )
        self.main_window.save_btn.setEnabled((has_assigned or has_assigned_aug) and not processing)

