import os
import cv2
from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QMutex, QMutexLocker, QRect, QSettings, QCoreApplication, QDir
from PyQt6.QtWidgets import QMessageBox # Use directly for icons
import numpy as np
from typing import List, Dict, Any, Optional

# Use local imports due to flat structure
from data_handler import DataHandler
from yolo_processor import YoloProcessor
from image_augmenter import ImageAugmenter
from main_window import MainWindow # Need main window type hint
from yolo_trainer import start_yolo_training # Import for training

# --- Worker Threads ---

class WorkerSignals(QObject):
    """Defines signals available from worker threads."""
    progress = pyqtSignal(int, int, str)  # current, total, status_message
    result = pyqtSignal(object)           # Emit results (type depends on worker)
    error = pyqtSignal(str)               # Emit error message
    finished = pyqtSignal()               # Emit when finished

    # Signals specific to training
    epochCompleted = pyqtSignal(int, int, dict) # current_epoch, total_epochs, metrics_dict
    trainingLogMessage = pyqtSignal(str)        # For relaying log messages from yolo_trainer

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


class SaveDatasetWorker(QThread):
    """Worker thread for augmenting (optional) and saving the dataset."""
    def __init__(self, data_handler: DataHandler, image_augmenter: ImageAugmenter,
                 output_dir: str, config: dict):
        super().__init__()
        self.signals = WorkerSignals()
        self.data_handler = data_handler
        self.image_augmenter = image_augmenter
        self.output_dir = output_dir
        self.train_split = config.get('train_split', 0.8)
        self.num_augmentations = config.get('augment_count', 0) # Get count from config
        self._is_running = True
        self.signals._error_emitted = False # Helper flag for progress callback
        self.current_step = 0 # Tracks current major step (augmentation or saving)
        self.total_steps = 0  # Tracks total major steps

    def run(self):
        self.total_steps = 1 # Start with 1 major step for saving
        self.current_step = 0
        annotated_image_paths = []

        try:
            # --- Step 0: Setup --- #
            def progress_callback(current_dh_item, total_dh_items, message): # current_dh_item, total_dh_items from DataHandler
                if not self._is_running: raise InterruptedError("Operation cancelled")
                
                progress_within_saving_step = 0.0
                if total_dh_items > 0:
                    progress_within_saving_step = float(current_dh_item) / total_dh_items
                
                # self.current_step is the major step index for saving phase (e.g., N_aug_images)
                # self.total_steps is total major steps (e.g., N_aug_images + 1)
                overall_progress_value = self.current_step + progress_within_saving_step
                
                # Emit progress scaled from 0 to total_steps*100
                self.signals.progress.emit(int(overall_progress_value * 100), self.total_steps * 100, message)


            # --- Step 1 (Optional): Augmentation --- #
            if self.num_augmentations > 0:
                self.data_handler.clear_augmented_images() # Clear previous augmentations
                annotated_image_paths = [
                    path for path, annotations in self.data_handler.annotations.items()
                    if any(ann.get('class_id', -1) >= 0 for ann in annotations)
                ]
                if not annotated_image_paths:
                    self.signals.error.emit("No images with assigned annotations to augment.")
                    self.signals._error_emitted = True
                    return

                total_aug_images = len(annotated_image_paths)
                self.total_steps += total_aug_images # Add augmentation major steps
                self.signals.progress.emit(0, self.total_steps * 100, f"Starting augmentation ({self.num_augmentations} per image)...")

                for i, image_path in enumerate(annotated_image_paths):
                    if not self._is_running: return
                    self.current_step = i # Current major step (0 to total_aug_images - 1)
                    self.signals.progress.emit(self.current_step * 100, self.total_steps * 100, f"Augmenting {i+1}/{total_aug_images}: {os.path.basename(image_path)}")

                    annotations = self.data_handler.annotations[image_path]
                    img_bgr = cv2.imread(image_path)

                    if img_bgr is None or not annotations:
                        print(f"Warning: Skipping augmentation for {image_path} (load fail or no annotations).")
                        continue

                    augmented_data = self.image_augmenter.augment_image(
                        img_bgr, annotations, self.num_augmentations
                    )

                    if augmented_data:
                        self.data_handler.add_augmented_images(image_path, augmented_data)
                
                self.current_step = total_aug_images # After augmentation, current_step is num of aug images done


            # --- Step 2: Save Dataset --- #
            # self.current_step is now effectively the starting point for the saving "major step"
            # e.g. if 3 aug images, self.current_step = 3. total_steps = 3 (aug) + 1 (save) = 4.
            # Saving phase contributes from step 3 to 4.
            if not self._is_running: return
            self.signals.progress.emit(self.current_step * 100, self.total_steps * 100, "Starting dataset save...")

            success = self.data_handler.save_yolo_dataset(
                self.output_dir,
                train_split=self.train_split,
                progress_callback=progress_callback
            )

            if not self._is_running: return

            if success:
                final_path = os.path.join(self.output_dir, 'yolo_dataset')
                self.signals.result.emit(final_path)
            else:
                 if not self.signals._error_emitted:
                      self.signals.error.emit("Failed to save dataset (unknown reason).")

        except InterruptedError:
             self.signals.error.emit("Dataset save cancelled.")
        except Exception as e:
            if not self.signals._error_emitted:
                 self.signals.error.emit(f"Save Dataset Error: {e}")
                 self.signals._error_emitted = True
            import traceback
            traceback.print_exc()
        finally:
            self.signals.finished.emit()

    def stop(self):
        self._is_running = False


class ProcessAllImagesWorker(QThread):
    """Worker thread for running object detection on all images."""
    def __init__(self, yolo_processor: YoloProcessor, image_paths: List[str], config: dict, data_handler: DataHandler):
        super().__init__()
        self.signals = WorkerSignals()
        self.yolo_processor = yolo_processor
        self.image_paths = image_paths
        self.conf_threshold = config.get('conf_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.data_handler = data_handler # To update annotations directly
        self._is_running = True

    def run(self):
        total_images = len(self.image_paths)
        if total_images == 0:
            self.signals.finished.emit()
            return

        for i, image_path in enumerate(self.image_paths):
            if not self._is_running:
                self.signals.error.emit("Processing all images cancelled.")
                break
            try:
                self.signals.progress.emit(i, total_images, f"Processing {i+1}/{total_images}: {os.path.basename(image_path)}...")

                detections = self.yolo_processor.detect(
                    image_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )

                if not self._is_running:  # Check again after potentially long detection
                    self.signals.error.emit(f"Processing cancelled during detection of {os.path.basename(image_path)}.")
                    break

                if detections is None:
                    # Emit an error for this specific image, but continue with others
                    self.signals.error.emit(f"Detection failed for {os.path.basename(image_path)}")
                    # Emit an empty result so UI might clear previous annotations if needed
                    # self.signals.result.emit({'image_path': image_path, 'detections': []}) # Or handle differently
                else:
                    # Update DataHandler immediately
                    self.data_handler.update_annotations_from_detections(image_path, detections)
                    self.signals.progress.emit(i + 1, total_images, f"Processed {os.path.basename(image_path)}, found {len(detections)} objects.")
                    # Emit result for this image so UI can update if it's the currently displayed one
                    self.signals.result.emit({'image_path': image_path, 'detections': detections})

            except Exception as e:
                self.signals.error.emit(f"Error processing {os.path.basename(image_path)}: {e}")
                # Optionally, decide if one error should stop the whole batch or just report and continue
        self.signals.finished.emit()

    def stop(self):
        self._is_running = False


class TrainingWorker(QThread):
    """Worker thread for running YOLO model training."""
    def __init__(self, training_params: dict):
        super().__init__()
        self.signals = WorkerSignals() # Reuses existing signals object, now with new signals
        self.training_params = training_params
        self._is_running = True # For potential cancellation (though ultralytics handles Ctrl+C)

    def run(self):
        try:
            if not self._is_running: return
            self.signals.trainingLogMessage.emit("Initializing training process...")

            # Define callbacks to pass to start_yolo_training
            def progress_cb(current_epoch, total_epochs, metrics):
                if not self._is_running: return # Check for cancellation
                self.signals.epochCompleted.emit(current_epoch, total_epochs, metrics)
            
            def log_cb(message):
                if not self._is_running: return
                self.signals.trainingLogMessage.emit(message)

            training_results = start_yolo_training(
                self.training_params,
                progress_callback=progress_cb,
                log_callback=log_cb
            )

            if not self._is_running:
                self.signals.error.emit("Training was cancelled by the user.")
            elif training_results.get("success"):
                self.signals.result.emit(training_results) # Pass the whole result dict
            else:
                error_msg = training_results.get("message", "Unknown training failure.")
                if training_results.get("error"): # Detailed error if available
                    error_msg += f"\nDetails: {training_results.get('error')}"
                self.signals.error.emit(error_msg)

        except Exception as e:
            import traceback
            self.signals.error.emit(f"Training Worker Error: {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

    def stop(self):
        # Note: Stopping an ongoing ultralytics training process externally is complex.
        # Ultralytics handles SIGINT (Ctrl+C) for graceful shutdown and saving.
        # A more robust solution would involve communicating with the ultralytics process
        # or using features they provide for early stopping, if available programmatically.
        # For now, this flag mainly prevents new signals if called before/between epochs.
        self._is_running = False
        self.signals.trainingLogMessage.emit("Training stop request received. Attempting to halt...")
        # How to effectively stop model.train() from here is the challenge.
        # If model.stop() or similar exists in ultralytics, call it here.


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
    projectStateChanged = pyqtSignal(str, bool) # project_path, is_dirty

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
        self.current_project_path: Optional[str] = None

        self._connect_signals()
        self._update_ui_state() # Initial UI state

        # Attempt to load last project on startup
        self._load_last_project()

    def _connect_signals(self):
        """Connect signals from UI and workers to logic slots."""
        mw = self.main_window
        mw.addImagesClicked.connect(self._on_add_images)
        mw.selectModelClicked.connect(self._on_select_model)
        mw.processImageClicked.connect(self._on_process_image)
        mw.processAllImagesClicked.connect(self._on_process_all_images_requested)
        mw.saveDatasetClicked.connect(self._on_save_dataset)
        mw.addClassClicked.connect(self._on_add_class)
        mw.removeClassClicked.connect(self._on_remove_class)
        mw.imageSelected.connect(self._on_image_selected_in_list)
        mw.classSelected.connect(self._on_class_selected_in_list)
        mw.configChanged.connect(self._on_config_changed)
        mw.drawBoxClicked.connect(self._on_draw_box_toggled)
        mw.deleteSelectedBoxClicked.connect(self._on_delete_selected_box) # From button
        mw.image_canvas.newBoxDrawn.connect(self._on_new_box_drawn)
        mw.image_canvas.boxSelected.connect(self._on_box_selected_in_canvas)
        mw.image_canvas.deleteSelectionKeyPressed.connect(self._on_delete_selected_box) # From keypress

        mw.trainModelParamsCollected.connect(self._on_train_model_request) # Connect training request

        # Add connections for project file operations (assuming MainWindow emits these)
        mw.openProjectRequested.connect(self.open_project)
        mw.saveProjectRequested.connect(self.save_project)
        mw.saveProjectAsRequested.connect(self.save_project_as)

        # Logic -> UI
        self.statusChanged.connect(mw.set_status)
        self.progressChanged.connect(mw.set_progress)
        self.imageDisplayRequested.connect(mw.display_image)
        self.annotationsChanged.connect(mw.update_annotations)
        self.classListChanged.connect(mw.update_class_names_display)
        self.projectStateChanged.connect(mw._update_window_title) # Connect to update window title
        # self.uiStateUpdated.connect(mw._update_buttons_state) # UI updates itself based on state

    # --- Project Operations ---
    def _load_last_project(self):
        settings = QSettings()
        last_project = settings.value("project/lastProjectPath", "", type=str)
        if last_project and os.path.isfile(last_project):
            print(f"Attempting to load last project: {last_project}")
            self.open_project(last_project)
        else:
            self._update_project_state() # Emit initial state (no project, not dirty)

    def _update_project_state(self):
        """Updates state variables and emits signal based on current project."""
        is_dirty = self.data_handler.is_dirty()
        project_path = self.current_project_path if self.current_project_path else "Unsaved Project"
        self.projectStateChanged.emit(project_path, is_dirty)
        self._update_ui_state() # Update button enabled states etc.

    @pyqtSlot(str)
    def open_project(self, file_path: str):
        if self._check_save_before_proceed():
            if self.data_handler.load_project(file_path):
                self.current_project_path = file_path
                settings = QSettings()
                settings.setValue("project/lastProjectPath", file_path)
                # Trigger UI refresh
                self._reload_ui_from_data()
                # Reload Yolo model if specified in project
                if self.data_handler.model_path and os.path.isfile(self.data_handler.model_path):
                    self.yolo_processor.load_model(self.data_handler.model_path)
                else: # No model in project or path invalid, clear current
                    self.yolo_processor.load_model(None) # Clear processor
                self.main_window.set_model_path(self.yolo_processor.model_path) # Update UI
                self.statusChanged.emit(f"Project '{os.path.basename(file_path)}' loaded.", 3000)
            else:
                self.statusChanged.emit(f"Failed to load project: {file_path}", 5000)
                # Clear potentially partially loaded state
                self.new_project() # This will call _check_save which might be redundant
            self._update_project_state()

    @pyqtSlot()
    def save_project(self):
        if self.current_project_path:
            if self.data_handler.save_project(self.current_project_path):
                self.statusChanged.emit(f"Project saved to '{os.path.basename(self.current_project_path)}'.", 3000)
                self._update_project_state()
                return True # Indicate success
            else:
                self.statusChanged.emit("Error saving project.", 5000)
                return False # Indicate failure
        else:
            # If no current path, treat as Save As
            return self.main_window._prompt_save_project_as() # Ask MainWindow to trigger the dialog

    @pyqtSlot(str)
    def save_project_as(self, file_path: str):
        if self.data_handler.save_project(file_path):
            self.current_project_path = file_path
            settings = QSettings()
            settings.setValue("project/lastProjectPath", file_path)
            self.statusChanged.emit(f"Project saved as '{os.path.basename(file_path)}'.", 3000)
            self._update_project_state()
            return True # Indicate success
        else:
            self.statusChanged.emit("Error saving project.", 5000)
            return False # Indicate failure

    def new_project(self):
        """Resets the application state for a new project."""
        if self._check_save_before_proceed():
            self.data_handler = DataHandler() # Create a fresh data handler
            self.yolo_processor.load_model(None) # Clear yolo processor
            self.current_project_path = None
            self.current_image_path = None
            # Reset UI elements
            self._reload_ui_from_data()
            self.statusChanged.emit("New project started.", 3000)
            self._update_project_state()
            return True
        return False

    def _check_save_before_proceed(self) -> bool:
        """Checks if the project is dirty and prompts the user to save. Returns False if cancelled."""
        if not self.data_handler.is_dirty():
            return True # Nothing to save

        project_name = os.path.basename(self.current_project_path) if self.current_project_path else "Unsaved Project"
        reply = QMessageBox.question(self.main_window,
                                     "Unsaved Changes",
                                     f"Project '{project_name}' has unsaved changes.\nDo you want to save them?",
                                     QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                     QMessageBox.StandardButton.Save)

        if reply == QMessageBox.StandardButton.Save:
            return self.save_project() # Returns True on success, False on failure/cancel
        elif reply == QMessageBox.StandardButton.Cancel:
            return False
        else: # Discard
            return True

    def _reload_ui_from_data(self):
        """Refreshes the UI completely based on the current DataHandler state."""
        # Clear existing UI elements
        self.main_window.images_list.clear()
        self.main_window.classes_list.clear()
        self.main_window.image_canvas.set_image(None) 
        self.main_window.update_annotations([]) 

        # Load new data
        self.main_window.add_images_to_list(self.data_handler.image_paths)
        self.main_window.update_class_names_display(self.data_handler.class_names)
        # Model path is set from data_handler when project loads, or by user interaction
        # YoloProcessor's model should align with data_handler.model_path
        # This is handled in open_project and _on_select_model
        self.main_window.set_model_path(self.data_handler.model_path)


        # Select first image if available
        if self.data_handler.image_paths:
            first_image = self.data_handler.image_paths[0]
            self.main_window.images_list.setCurrentRow(0) 
        else:
            self.current_image_path = None
            self.imageDisplayRequested.emit(None)
            self.main_window.update_annotations([])

        self._update_project_state()

    def get_current_dataset_yaml_path(self) -> Optional[str]:
        """Returns the expected path to data.yaml if the dataset has been saved."""
        # This path depends on where DataHandler.save_yolo_dataset places it.
        # Assuming it's in <output_dir_chosen_by_user>/yolo_dataset/dataset.yaml
        # We need to know the last output_dir used for saving.
        # DataHandler doesn't directly store this; SaveDatasetWorker receives it.
        # For simplicity, we might need DataHandler to store the last successful save_dir.
        # Or, we retrieve it from settings if we save it there after a successful dataset save.

        # Let's assume DataHandler can provide the root of the last saved dataset.
        if self.data_handler and hasattr(self.data_handler, 'last_saved_dataset_root_dir') and self.data_handler.last_saved_dataset_root_dir:
            yaml_path = os.path.join(self.data_handler.last_saved_dataset_root_dir, "data.yaml")
            if os.path.isfile(yaml_path):
                return yaml_path
        # Fallback: check settings if we store it there (not currently implemented this way)
        # last_dataset_output_dir = QSettings().value("dataset/lastOutputDir", None)
        # if last_dataset_output_dir:
        #    yaml_path = os.path.join(last_dataset_output_dir, "yolo_dataset", "dataset.yaml")
        #    if os.path.isfile(yaml_path):
        #        return yaml_path
        return None

    def _start_worker(self, worker_class, *args):
        """Helper to start a worker thread and manage state."""
        if self._is_processing:
            self.statusChanged.emit("Busy: Another process is running.", 3000)
            return

        self._is_processing = True
        self._update_ui_state(processing=True) # Disable controls

        self._current_worker = worker_class(*args)
        # Generic signals
        self._current_worker.signals.progress.connect(self._handle_worker_progress)
        self._current_worker.signals.result.connect(self._handle_worker_result)
        self._current_worker.signals.error.connect(self._handle_worker_error)
        self._current_worker.signals.finished.connect(self._handle_worker_finished)

        # Training specific signals - connect only if they exist on the worker's signals object
        if hasattr(self._current_worker.signals, 'epochCompleted') and hasattr(self._current_worker.signals.epochCompleted, 'connect'):
            self._current_worker.signals.epochCompleted.connect(self._handle_training_epoch_completed)
        if hasattr(self._current_worker.signals, 'trainingLogMessage') and hasattr(self._current_worker.signals.trainingLogMessage, 'connect'):
            self._current_worker.signals.trainingLogMessage.connect(self._handle_training_log_message)

        self._current_worker.start()



    @pyqtSlot(list)
    def _on_add_images(self, file_paths: List[str]):
        """Handles adding images triggered by the UI."""
        if file_paths:
            added_count = self.data_handler.add_images(file_paths)
            self.main_window.add_images_to_list(file_paths) 
            self.statusChanged.emit(f"Added {added_count} new images.", 3000)
            if self.data_handler.is_dirty(): self._update_project_state()
            if added_count > 0 and self.main_window.images_list.count() == added_count:
                 self.main_window.images_list.setCurrentRow(0)

    @pyqtSlot(str)
    def _on_select_model(self, model_path: Optional[str]):
        """Handles selecting a model triggered by the UI."""
        
        # Case 1: Clearing the model (model_path is None or empty string from dialog cancel)
        if not model_path:
            if self.data_handler.model_path is not None: # If a model was actually set before
                self.data_handler.set_model_path(None) # Clears in DH, sets dirty
                self.yolo_processor.load_model(None)   # Clears processor state
                self.main_window.set_model_path(None)  # Clears UI
                self.statusChanged.emit("Model selection cleared.", 3000)
                self._update_project_state() # Reflect dirty state change
            return

        # Case 2: Setting a new model (model_path is a non-empty string)
        # DataHandler.set_model_path checks os.path.isfile. If it fails, model_path is not set in DH.
        if not self.data_handler.set_model_path(model_path):
            # data_handler.set_model_path returned False, meaning model_path is not a valid file.
            # self.data_handler.model_path remains unchanged (or None if it was None).
            self.statusChanged.emit(f"Invalid model file: {model_path}", 5000)
            self.main_window.show_message("Model Error", f"The specified model path is not a valid file: {model_path}", level="warning")
            # Ensure UI reflects the actual model in data_handler (which might be the previous one or None)
            self.main_window.set_model_path(self.data_handler.model_path)
            return

        # At this point, data_handler.model_path IS model_path, and it's a file.
        # Now, try to load it with YoloProcessor
        if self.yolo_processor.load_model(model_path):
            # Successfully loaded by YoloProcessor
            self.main_window.set_model_path(model_path) # Update UI
            self.statusChanged.emit(f"Model selected: {os.path.basename(model_path)}", 3000)
            # data_handler.set_model_path already marked dirty if it changed.
            self._update_project_state() # Update UI based on dirty state, etc.
        else:
            # YoloProcessor failed to load the model, even though it's a file.
            self.statusChanged.emit(f"Failed to load model: {model_path}", 5000)
            self.main_window.show_message("Model Error", f"Could not load model file (e.g., corrupted or incompatible): {model_path}", level="warning")
            
            # Revert: Clear the model in DataHandler, YoloProcessor (already cleared by failed load_model), and UI
            # data_handler.model_path is currently model_path. Change it back to None.
            self.data_handler.set_model_path(None) # This will set dirty if model_path was not None
            # yolo_processor.load_model(None) # YoloProcessor.load_model(model_path) failing should clear its state.
            self.main_window.set_model_path(None)  # Clear UI
            self._update_project_state() # Reflect changes


    @pyqtSlot()
    def _on_process_image(self):
        """Initiates object detection on the currently selected image."""
        if not self.current_image_path:
            self.statusChanged.emit("No image selected.", 3000)
            return
        if not self.yolo_processor.is_model_loaded():
            self.statusChanged.emit("No model selected or loaded.", 3000)
            self.main_window.show_message("Processing Error", "Please select a valid YOLO model file first.", level="warning")
            return

        config = self.main_window.get_config() # Get current conf/iou
        self._start_worker(DetectionWorker, self.yolo_processor, self.current_image_path, config)

    @pyqtSlot()
    def _on_process_all_images_requested(self):
        """Initiates object detection on all loaded images."""
        if not self.data_handler.image_paths:
            self.statusChanged.emit("No images loaded to process.", 3000)
            return
        if not self.yolo_processor.is_model_loaded():
            self.statusChanged.emit("No model selected or loaded.", 3000)
            self.main_window.show_message("Processing Error", "Please select a valid YOLO model file first.", level="warning")
            return

        config = self.main_window.get_config()
        all_image_paths = list(self.data_handler.image_paths) # Get a copy

        self.statusChanged.emit(f"Starting processing for {len(all_image_paths)} images...", 0)
        self._start_worker(ProcessAllImagesWorker, self.yolo_processor, all_image_paths, config, self.data_handler)

    @pyqtSlot(str)
    def _on_save_dataset(self, output_dir: str):
        """Initiates augmenting (if count > 0) and saving the dataset."""
        if not self.data_handler.class_names:
             self.main_window.show_message("Save Error", "Cannot save dataset without defined classes.", level="warning")
             self.statusChanged.emit("Define classes before saving.", 3000)
             return

        config = self.main_window.get_config()
        num_augmentations = config.get('augment_count', 0)

        if num_augmentations > 0 and not any(any(ann.get('class_id', -1) >= 0 for ann in anns) for anns in self.data_handler.annotations.values()):
             self.main_window.show_message("Augmentation Warning", "No annotations with assigned classes found. Augmentation will be skipped.", level="warning")
        elif num_augmentations == 0 and not any(any(ann.get('class_id', -1) >= 0 for ann in anns) for anns in self.data_handler.annotations.values()):
             self.main_window.show_message("Save Warning", "No annotations with assigned classes found. The dataset might be empty or incomplete.", level="info")

        self.statusChanged.emit("Starting dataset save...", 0)
        self._start_worker(SaveDatasetWorker, self.data_handler, self.image_augmenter, output_dir, config)

    @pyqtSlot(str)
    def _on_add_class(self, class_name: str):
        """Handles adding a class triggered by the UI."""
        if self.data_handler.add_class(class_name):
            self.classListChanged.emit(self.data_handler.class_names)
            self.statusChanged.emit(f"Class '{class_name}' added.", 2000)
            if self.data_handler.is_dirty(): self._update_project_state()
        else:
            self.statusChanged.emit(f"Class '{class_name}' already exists or is invalid.", 3000)

    @pyqtSlot(str)
    def _on_remove_class(self, class_name: str):
        """Handles removing a class triggered by the UI."""
        if self.data_handler.remove_class(class_name):
            self.classListChanged.emit(self.data_handler.class_names)
            if self.current_image_path:
                 current_annotations = self.data_handler.annotations.get(self.current_image_path, [])
                 self.annotationsChanged.emit(current_annotations)
            self.statusChanged.emit(f"Class '{class_name}' removed.", 2000)
            if self.data_handler.is_dirty(): self._update_project_state()
        else:
            self.statusChanged.emit(f"Failed to remove class '{class_name}'.", 3000)

    @pyqtSlot(str)
    def _on_image_selected_in_list(self, image_path: str):
        """Handles selecting an image in the UI list."""
        if image_path and image_path != self.current_image_path:
            self.current_image_path = image_path
            self.imageDisplayRequested.emit(image_path)
            current_annotations = self.data_handler.annotations.get(image_path, [])
            self.annotationsChanged.emit(current_annotations)
            self.main_window.image_canvas.select_box(-1)
            self._update_ui_state() 
            self.statusChanged.emit(f"Selected: {os.path.basename(image_path)}", 0)
        elif not image_path:
             self.current_image_path = None
             self.imageDisplayRequested.emit(None)
             self.main_window.update_annotations([]) 
             self._update_ui_state()
             self.statusChanged.emit("No image selected.", 0)


    @pyqtSlot(str)
    def _on_class_selected_in_list(self, class_name: str):
        """Handles selecting a class in the UI list."""
        selected_box_index = self.main_window.image_canvas.selected_box_idx
        if self.current_image_path and selected_box_index != -1 and class_name:
            try:
                class_id = self.data_handler.class_names.index(class_name)
                if self.data_handler.assign_class_to_box(self.current_image_path, selected_box_index, class_id):
                    current_annotations = self.data_handler.annotations.get(self.current_image_path, [])
                    self.annotationsChanged.emit(current_annotations)
                    self.statusChanged.emit(f"Assigned class '{class_name}' to selected box.", 2000)
                    if self.data_handler.is_dirty(): self._update_project_state()
                else:
                     self.statusChanged.emit(f"Failed to assign class '{class_name}'.", 3000)
            except ValueError:
                self.statusChanged.emit(f"Class '{class_name}' not found.", 3000)
        elif class_name:
             self.statusChanged.emit(f"Selected class: {class_name}", 0)


    @pyqtSlot(int)
    def _on_box_selected_in_canvas(self, box_index: int):
         """Handles selecting a box directly on the canvas."""
         selected_class_item = self.main_window.classes_list.currentItem()
         if self.current_image_path and box_index != -1 and selected_class_item:
             class_name = selected_class_item.text()
             try:
                 class_id = self.data_handler.class_names.index(class_name)
                 if self.data_handler.assign_class_to_box(self.current_image_path, box_index, class_id):
                     current_annotations = self.data_handler.annotations.get(self.current_image_path, [])
                     self.annotationsChanged.emit(current_annotations)
                     self.statusChanged.emit(f"Assigned class '{class_name}' to selected box.", 2000)
                     if self.data_handler.is_dirty(): self._update_project_state()
                 else:
                      self.statusChanged.emit(f"Failed to assign class '{class_name}'.", 3000)
             except ValueError:
                 self.statusChanged.emit(f"Class '{class_name}' not found.", 3000)
         elif box_index != -1:
             self.statusChanged.emit(f"Box {box_index} selected.", 0)
             # Optional: Highlight class in list if box has one
             # annotation = self.data_handler.annotations[self.current_image_path][box_index]
             # class_id = annotation.get('class_id', -1)
             # if class_id != -1 and class_id < len(self.data_handler.class_names):
             #    self.main_window.select_class_in_list(self.data_handler.class_names[class_id])


    @pyqtSlot(dict)
    def _on_config_changed(self, config: dict):
        """Handles changes in configuration (confidence, IoU, etc.)."""
        self.current_config = config
        self.statusChanged.emit(f"Config updated: Conf={config['conf_threshold']:.2f}, IoU={config['iou_threshold']:.2f}", 2000)

    @pyqtSlot(bool)
    def _on_draw_box_toggled(self, enabled: bool):
        """Handles toggling the draw box mode in the canvas."""
        self.main_window.image_canvas.set_drawing_enabled(enabled)
        status = "Draw Box mode enabled." if enabled else "Draw Box mode disabled."
        self.statusChanged.emit(status, 2000)
        if enabled:
            self.main_window.image_canvas.select_box(-1) 
            self.main_window.classes_list.clearSelection() 

    @pyqtSlot(QRect)
    def _on_new_box_drawn(self, box_rect_img: QRect):
        """Handles a new box being drawn on the canvas."""
        if self.current_image_path:
            new_box_index = self.data_handler.add_manual_box(self.current_image_path, box_rect_img)
            if new_box_index != -1:
                current_annotations = self.data_handler.annotations.get(self.current_image_path, [])
                self.annotationsChanged.emit(current_annotations)
                self.main_window.image_canvas.select_box(new_box_index)
                self.statusChanged.emit("New box added.", 2000)
                if self.data_handler.is_dirty(): self._update_project_state()
            else:
                 self.statusChanged.emit("Failed to add new box.", 3000)
        else:
             self.statusChanged.emit("Select an image before drawing boxes.", 3000)


    @pyqtSlot()
    def _on_delete_selected_box(self):
        """Handles deleting the currently selected box (from button or keypress)."""
        selected_box_index = self.main_window.image_canvas.selected_box_idx
        if self.current_image_path and selected_box_index != -1:
            if self.data_handler.delete_box(self.current_image_path, selected_box_index):
                current_annotations = self.data_handler.annotations.get(self.current_image_path, [])
                self.annotationsChanged.emit(current_annotations)
                self.main_window.image_canvas.select_box(-1) 
                self.statusChanged.emit("Selected box deleted.", 2000)
                if self.data_handler.is_dirty(): self._update_project_state()
            else:
                 self.statusChanged.emit("Failed to delete selected box.", 3000)
        else:
             self.statusChanged.emit("No box selected to delete.", 3000)

    @pyqtSlot(dict)
    def _on_train_model_request(self, params: dict):
        """Handles the request to start model training with the given parameters."""
        if self._is_processing:
            self.statusChanged.emit("Busy: Another process is running.", 3000)
            self.main_window.show_message("Training Busy", "Another operation (e.g., detection, saving, or training) is already in progress.", "warning")
            return

        yaml_path_from_params = params.get("data")
        if not yaml_path_from_params:
            self.statusChanged.emit("Dataset YAML path not provided in parameters.", 4000)
            self.main_window.show_message("Training Error", "Dataset YAML path not provided.", "error")
            return

        # Ensure the YAML path is absolute before using it
        if not os.path.isabs(yaml_path_from_params):
            # If relative, os.path.abspath will resolve it against the current working directory.
            resolved_yaml_path = os.path.abspath(yaml_path_from_params)
        else:
            resolved_yaml_path = yaml_path_from_params
        
        # Verify the resolved path points to an existing file
        if not os.path.isfile(resolved_yaml_path):
            self.statusChanged.emit(f"Dataset YAML not found or is invalid: {resolved_yaml_path}", 4000)
            self.main_window.show_message("Training Error", f"Dataset configuration file (data.yaml) not found or is invalid: {resolved_yaml_path}. Ensure the dataset is saved correctly.", "error")
            return
        
        # Update params with the (potentially now absolute) path
        params["data"] = resolved_yaml_path 
        
        # Potentially add more validation here if needed before starting worker
        self.statusChanged.emit(f"Starting training with model: {params.get('model')} using {os.path.basename(resolved_yaml_path)}...", 0)
        # The _start_worker method will set _is_processing = True and update UI state
        self._start_worker(TrainingWorker, params)


    # --- Worker Signal Handlers ---

    @pyqtSlot(int, int, str)
    def _handle_worker_progress(self, current: int, total: int, message: str):
        """Updates the progress bar and status bar."""
        self.progressChanged.emit(current, total)
        self.statusChanged.emit(message, 0)

    @pyqtSlot(object)
    def _handle_worker_result(self, result):
        """Handles results from worker threads."""
        worker_type = type(self._current_worker)

        if worker_type is DetectionWorker:
            image_path = result['image_path']
            detections = result['detections']
            self.data_handler.update_annotations_from_detections(image_path, detections)
            if image_path == self.current_image_path:
                self.annotationsChanged.emit(self.data_handler.annotations[image_path])
            self.statusChanged.emit(f"Processing finished for {os.path.basename(image_path)}. {len(detections)} boxes found.", 3000)
            if self.data_handler.is_dirty(): self._update_project_state()

        elif worker_type is ProcessAllImagesWorker: # Handle results from the new worker
            image_path = result['image_path']
            # Detections already processed and data_handler updated by the worker itself.
            if image_path == self.current_image_path:
                # If the currently viewed image was processed, update its annotations display
                self.annotationsChanged.emit(self.data_handler.annotations.get(image_path, []))
            if self.data_handler.is_dirty(): self._update_project_state()

        elif worker_type is SaveDatasetWorker:
             dataset_path = result # This is the .../yolo_dataset directory
             print(f"[AppLogic Debug] SaveDatasetWorker result (dataset_path): {dataset_path}") # DEBUG
             if self.data_handler and os.path.isdir(dataset_path):
                 self.data_handler.last_saved_dataset_root_dir = dataset_path 
                 print(f"[AppLogic Debug] Set data_handler.last_saved_dataset_root_dir to: {self.data_handler.last_saved_dataset_root_dir}") # DEBUG
             else:
                 print(f"[AppLogic Debug] Did NOT set data_handler.last_saved_dataset_root_dir. data_handler: {self.data_handler}, os.path.isdir(dataset_path): {os.path.isdir(dataset_path) if dataset_path else 'N/A'}") # DEBUG
             num_aug = self.current_config.get('augment_count', 0)
             msg = f"Dataset saved successfully to {dataset_path}"
             if num_aug > 0:
                  msg += f" (including {num_aug} augmentation(s) per image)."
             else:
                  msg += "."
             self.statusChanged.emit(msg, 5000)
             self.main_window.show_message("Save Successful", f"YOLO dataset saved to:\n{dataset_path}", level="info")
             self._update_ui_state() # Update UI as save dataset might enable training button

        elif worker_type is TrainingWorker:
            success = result.get("success", False)
            message = result.get("message", "Training finished with unknown status.")
            best_model_path = result.get("best_model_path")
            self.statusChanged.emit(message, 7000)
            if success and best_model_path:
                self.main_window.show_message("Training Successful", f"{message}\nBest model saved to: {best_model_path}", "info")
                # Optional: Offer to load this model as the current detection model
                reply = QMessageBox.question(self.main_window, "Load Trained Model?", 
                                             f"Training successful. Best model saved to:\n{best_model_path}\n\nWould you like to load this model for detection?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.Yes)
                if reply == QMessageBox.StandardButton.Yes:
                    self._on_select_model(best_model_path)
            elif success: # Success but no model path for some reason
                self.main_window.show_message("Training Finished", message, "info")
            else: # Not successful
                detailed_error = result.get("error", "")
                self.main_window.show_message("Training Failed", f"{message}\n{detailed_error}", "error")

        else:
             print(f"Warning: Received result from unknown worker type: {worker_type}")


    @pyqtSlot(str)
    def _handle_worker_error(self, error_msg: str):
        """Displays errors from worker threads."""
        print(f"Worker Error: {error_msg}")
        self.statusChanged.emit(f"Error: {error_msg}", 5000)
        self.main_window.show_message("Worker Error", error_msg, level="error")

    @pyqtSlot()
    def _handle_worker_finished(self):
        """Cleans up after a worker thread finishes."""
        self.statusChanged.emit("Ready", 0) 
        self.progressChanged.emit(0, 100) 
        self._is_processing = False
        self._current_worker = None
        self._update_ui_state(processing=False)

    def _update_ui_state(self, processing: Optional[bool] = None):
        """Updates the enabled/disabled state of UI elements based on app state."""
        if processing is None:
            processing = self._is_processing

        has_images = bool(self.data_handler.image_paths)
        has_model = self.yolo_processor.is_model_loaded()
        image_selected = self.current_image_path is not None
        box_selected = image_selected and self.main_window.image_canvas.selected_box_idx != -1
        has_classes = bool(self.data_handler.class_names) # Still useful for general state
        class_selected = self.main_window.classes_list.currentItem() is not None
        is_drawing = self.main_window.image_canvas._is_drawing_enabled
        is_dirty = self.data_handler.is_dirty()
        
        # Check if a valid dataset yaml path is available for training
        dataset_yaml_path = self.get_current_dataset_yaml_path()
        # Training is possible if not processing, yaml exists, and implies classes are defined.
        can_train = not processing and dataset_yaml_path is not None and os.path.isfile(dataset_yaml_path)

        state = {
            'is_processing': processing, 
            'enable_add_images': not processing,
            'enable_select_model': not processing,
            'enable_process': not processing and has_model and image_selected,
            'enable_process_all': not processing and has_model and has_images,
            'enable_save_dataset': not processing and has_images and has_classes, # Keep has_classes here, good check before saving.
            'enable_add_class': not processing,
            'enable_remove_class': not processing and class_selected,
            'enable_canvas_interaction': not processing and image_selected,
            'enable_delete_box': not processing and box_selected and not is_drawing,
            'enable_draw_box': not processing and image_selected,
            'enable_train_model': can_train, 
            'enable_save_project': not processing and (is_dirty or not self.current_project_path),
            'enable_save_project_as': not processing and has_images, 
            'enable_close_project': not processing, 
        }
        self.main_window._update_buttons_state(state)

    @pyqtSlot(int, int, dict)
    def _handle_training_epoch_completed(self, current_epoch: int, total_epochs: int, metrics: dict):
        """Handles epoch completion signals from TrainingWorker."""
        self.progressChanged.emit(current_epoch, total_epochs)
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (float, int))])
        self.statusChanged.emit(f"Epoch {current_epoch}/{total_epochs} | {metrics_str}", 0) if metrics_str else self.statusChanged.emit(f"Epoch {current_epoch}/{total_epochs}", 0)

    @pyqtSlot(str)
    def _handle_training_log_message(self, message: str):
        """Handles log messages from TrainingWorker."""
        print(f"[Train Log] {message}") # Print to console for now
        # Show important messages or errors in status bar
        if "TRAINING ERROR" in message.upper() or "TRACEBACK" in message.upper():
            short_msg = message.splitlines()[0]
            self.statusChanged.emit(f"Train Log: {short_msg}", 6000) 
        elif message.startswith("Epoch") and "Summary:" in message:
             self.statusChanged.emit(message, 0) # Keep epoch summaries in status if concise
        elif "Training started" in message or "Training finished" in message or "Output will be saved" in message or "Best model:" in message:
            self.statusChanged.emit(message, 3000)