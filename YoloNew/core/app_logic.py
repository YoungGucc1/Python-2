from PyQt6.QtCore import QObject, QThreadPool, QRect, QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox # For file dialogs
from typing import List, Dict, Optional, Tuple  # Add typing imports
import cv2
import os
import random
import copy

# Import your core components
from core.models import AppData, ImageAnnotation, BoundingBox
from core.data_handler import DataHandler
from core.yolo_processor import YoloProcessor
from core.image_augmenter import ImageAugmenter
from core.workers import DetectionWorker, AugmentationWorker  # Import AugmentationWorker
from core.state_manager import StateManager
from core import utils
from core import formats

class AppLogic(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.app_data = AppData()
        self.data_handler = DataHandler(self.app_data) # Pass shared data object
        self.yolo_processor = YoloProcessor()
        self.image_augmenter = ImageAugmenter()
        
        # Initialize instance variables
        self.current_image_path: str | None = None
        self.selected_box_canvas_index: int = -1 # Track selection in canvas
        
        # Initialize the state manager with a 30-second auto-save interval
        self.state_manager = StateManager(auto_save_interval=30)
        self.state_manager.set_app_data(self.app_data)
        
        # Set up a timer for auto-saving
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._check_auto_save)
        self.auto_save_timer.start(5000)  # Check every 5 seconds
        
        self.thread_pool = QThreadPool()
        print(f"Using max {self.thread_pool.maxThreadCount()} threads.")

        self._connect_ui_signals()
        
        # Load previous state if available - do this after all initialization
        self._load_previous_state()

    def _connect_ui_signals(self):
        # Connect signals FROM MainWindow TO AppLogic methods
        self.main_window.add_images_requested.connect(self.add_images)
        self.main_window.select_model_requested.connect(self.select_model)
        self.main_window.process_images_requested.connect(self.run_detection)
        self.main_window.save_dataset_requested.connect(self.save_dataset)
        self.main_window.save_state_requested.connect(self._on_save_state_requested)
        self.main_window.clear_state_requested.connect(self._on_clear_state_requested)
        self.main_window.image_selected.connect(self.load_image_and_annotations)
        self.main_window.delete_image_requested.connect(self._on_delete_image_requested)
        self.main_window.clear_images_requested.connect(self._on_clear_images_requested)
        self.main_window.import_classes_requested.connect(self._on_import_classes_requested)
        # Export classes is handled entirely in MainWindow
        self.main_window.class_added_requested.connect(self.add_class)
        self.main_window.class_removed_requested.connect(self.remove_class)
        self.main_window.class_selected_for_assignment.connect(self.assign_class_to_selected_box)

    # --- Methods Triggered by UI ---

    def _check_auto_save(self):
        """Called by timer to check if auto-save is needed"""
        if self.state_manager.auto_save_if_needed():
            # Show auto-save indicator if auto-save occurred
            self.main_window.show_auto_save_indicator()
        
    def _load_previous_state(self):
        """Load the previous application state if available"""
        if self.state_manager.load_state():
            print("Previous session state loaded successfully")
            
            # Clean up app_data by removing references to non-existent files and augmented images
            # that we don't want to keep between sessions
            paths_to_remove = []
            for path, annot in self.app_data.images.items():
                # Remove augmented images or images that don't exist on disk
                if annot.augmented_from is not None or not os.path.exists(path):
                    paths_to_remove.append(path)
                    
            # Remove the identified paths
            for path in paths_to_remove:
                del self.app_data.images[path]
                
            print(f"Cleaned up {len(paths_to_remove)} images (augmented or missing from disk)")
            
            # Update UI with loaded state data - already filtered by _get_original_image_paths
            image_paths = self._get_original_image_paths()
            self.main_window.update_image_list(image_paths)
            self.main_window.update_class_list(self.app_data.classes)
            
            # Update model label if a model was loaded
            if self.app_data.model_path:
                self.main_window.set_model_label(self.app_data.model_path)
                
            # Update button states based on loaded data
            self.update_button_states()
            
            # Select first image if available - do this after updating the image list
            if image_paths:
                try:
                    self.main_window.image_list_widget.setCurrentRow(0)
                    # This will trigger load_image_and_annotations via the selection changed signal
                except Exception as e:
                    print(f"Error selecting first image: {e}")
                
            # Show status message
            self.main_window.status_bar.showMessage("Previous session state loaded", 3000)
        else:
            print("No previous session state found or failed to load")
            
    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self.main_window,
            "Select Images",
            "", # Start directory
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if files:
            added = self.data_handler.add_image_paths(files)
            # Only show original images in the UI list, not augmented versions
            self.main_window.update_image_list(self._get_original_image_paths())
            if added and not self.current_image_path:
                # Select the first added image if none is selected
                 self.main_window.image_list_widget.setCurrentRow(0)
                 # load_image_and_annotations will be called by the selection change signal
                 
            # Save state after adding images
            self.state_manager.save_state()

    def select_model(self):
        file, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select YOLO Model",
            "",
            "PyTorch Models (*.pt);;ONNX Models (*.onnx)" # Adapt as needed
        )
        if file:
            self.app_data.model_path = file
            self.main_window.set_model_label(file)
            # Enable process button if images are loaded
            self.update_button_states()
            
            # Save state after selecting model
            self.state_manager.save_state()
            
    def load_image_and_annotations(self, image_path: str):
        self.current_image_path = image_path
        self.selected_box_canvas_index = -1 # Reset box selection
        if image_path in self.app_data.images:
             annotation_data = self.app_data.images[image_path]
             canvas = self.main_window.get_image_canvas()
             canvas.set_image(image_path)
             # Ensure image dimensions are loaded/stored in ImageAnnotation
             if annotation_data.width == 0 or annotation_data.height == 0:
                 h, w = canvas.cv_image.shape[:2]
                 annotation_data.width = w
                 annotation_data.height = h
             canvas.set_annotations(annotation_data.boxes, self.app_data.classes)
             self.update_button_states() # Processing may depend on current image
        else:
            print(f"Error: {image_path} not found in app data.")
            # Clear the canvas if image not found
            self.main_window.get_image_canvas().clear()

    def run_detection(self, image_paths: list):
        if not self.app_data.model_path:
            self.main_window.show_message("Error", "Please select a YOLO model first.", QMessageBox.Icon.Warning)
            return
        if not image_paths:
             self.main_window.show_message("Error", "No images selected/loaded for processing.", QMessageBox.Icon.Warning)
             return

        if not self.yolo_processor.is_model_loaded():
             # Try loading model now if not already loaded
             try:
                 self.yolo_processor.load_model(self.app_data.model_path)
             except Exception as e:
                  self.main_window.show_message("Error", f"Failed to load model:\n{e}", QMessageBox.Icon.Critical)
                  return

        self.main_window.set_ui_busy(True, f"Running detection on {len(image_paths)} image(s)...")

        # --- Worker Thread ---
        worker = DetectionWorker(self.yolo_processor, image_paths, self.app_data.images)
        # Connect signals from worker to AppLogic slots
        worker.signals.result.connect(self._handle_detection_result)
        worker.signals.finished.connect(self._on_detection_finished)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.progress.connect(self._update_worker_progress) # TODO: Implement progress update

        self.thread_pool.start(worker)

    def _run_augmentation(self, num_augmentations: int):
        """Run augmentation and return the results (now an internal method)"""
        annotated_images = self.data_handler.get_annotated_image_paths()
        if not annotated_images:
            self.main_window.show_message("Warning", "No annotated images found to augment.", QMessageBox.Icon.Warning)
            return None
        
        # Get original annotations
        original_annotations = {p: self.app_data.images[p] for p in annotated_images}
        
        # Directly call the augmenter instead of using a worker
        try:
            self.main_window.set_ui_busy(True, f"Augmenting {len(annotated_images)} images with {num_augmentations} variations each...")
            
            # Use our enhanced ImageAugmenter to create augmentations
            augmented_data = self.image_augmenter.augment_batch(original_annotations, num_augmentations)
            
            if not augmented_data:
                self.main_window.show_message("Warning", "No augmentations were successfully created.", QMessageBox.Icon.Warning)
                return None
            
            # Update progress
            self.main_window.status_bar.showMessage(f"Created {len(augmented_data)} augmented images", 3000)
            
            # Add the augmented data to the app_data
            self.data_handler.add_augmented_data(augmented_data)

            # Make sure UI only shows original images - no need to update UI here,
            # as this will be handled by _on_augmentation_finished
            
            return augmented_data
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Augmentation error: {str(e)}\n{error_details}")
            self.main_window.show_message("Error", f"Augmentation failed: {str(e)}", QMessageBox.Icon.Critical)
            return None

    def save_dataset(self, format_type: str, do_augment: bool, num_augmentations: int):
        """Combined save dataset and augment functionality"""
        # --- Get Save Options ---
        output_dir = QFileDialog.getExistingDirectory(self.main_window, "Select Output Directory")
        if not output_dir:
            return
            
        if not self.data_handler.has_annotations():
            self.main_window.show_message("Warning", "No annotations found to save.", QMessageBox.Icon.Warning)
            return
            
        # Set UI to busy state
        self.main_window.set_ui_busy(True, "Preparing dataset...")
        
        # Run augmentation if requested, using threaded approach
        if do_augment and num_augmentations > 0:
            self.main_window.set_ui_busy(True, f"Augmenting images with {num_augmentations} variations each...")
            
            # Get all annotated images
            annotated_images = self.data_handler.get_annotated_image_paths()
            if not annotated_images:
                self.main_window.show_message("Warning", "No annotated images found to augment.", QMessageBox.Icon.Warning)
                self.main_window.set_ui_busy(False)
                return
                
            # Pass only necessary data (paths, annotations)
            original_annotations = {p: self.app_data.images[p] for p in annotated_images}
            
            # Create worker for augmentation
            worker = AugmentationWorker(self.image_augmenter, original_annotations, num_augmentations)
            
            # Connect signals
            worker.signals.result.connect(lambda augmented_data: self._continue_save_dataset(augmented_data, format_type, output_dir, num_augmentations))
            worker.signals.error.connect(self._handle_worker_error)
            worker.signals.progress.connect(self._update_worker_progress)
            
            # Start worker
            self.thread_pool.start(worker)
        else:
            # No augmentation, proceed directly to saving
            self._continue_save_dataset(None, format_type, output_dir, num_augmentations)

    def _continue_save_dataset(self, augmented_data, format_type, output_dir, num_augmentations):
        """Continue saving dataset after augmentation is complete or skipped"""
        # Apply augmentation results if they exist
        if augmented_data:
            self.data_handler.add_augmented_data(augmented_data)
            # Progress update
            self.main_window.status_bar.showMessage(f"Created {len(augmented_data)} augmented images", 3000)
            
        # Now save the dataset (original + any augmented images)
        self.main_window.set_ui_busy(True, f"Saving dataset in {format_type.upper()} format...")
        
        try:
            # Get all annotated images - including augmented ones
            # Use get_annotated_image_paths directly instead of _get_original_image_paths
            # to include augmented images in the export
            annotated_paths = self.data_handler.get_annotated_image_paths()
            if not annotated_paths:
                self.main_window.show_message("Warning", "No annotations found to save after processing.", QMessageBox.Icon.Warning)
                self.main_window.set_ui_busy(False)
                return
            
            # Print some information about what's being saved
            orig_paths = [path for path, annot in self.app_data.images.items() 
                          if annot.augmented_from is None and os.path.exists(path) and 
                          annot.boxes and any(box.class_id >= 0 for box in annot.boxes)]
            aug_paths = [path for path in annotated_paths if path not in orig_paths]
            print(f"Saving dataset with {len(orig_paths)} original images and {len(aug_paths)} augmented images")
                
            # Create train/val split (80/20 default)
            train_split = 0.8
            random.shuffle(annotated_paths)
            split_idx = int(len(annotated_paths) * train_split)
            train_paths = annotated_paths[:split_idx]
            val_paths = annotated_paths[split_idx:]
            
            # Make sure we have at least one image in each split
            if not train_paths and val_paths:
                train_paths = [val_paths[0]]
                val_paths = val_paths[1:] if len(val_paths) > 1 else []
            elif not val_paths and train_paths:
                val_paths = [train_paths[-1]]
                train_paths = train_paths[:-1] if len(train_paths) > 1 else []
                
            # Call the appropriate format saver
            if format_type.lower() == 'yolo':
                formats.save_yolo(self.app_data, output_dir, train_paths, val_paths)
            elif format_type.lower() == 'coco':
                formats.save_coco(self.app_data, output_dir, train_paths, val_paths)
            elif format_type.lower() == 'voc':
                formats.save_voc(self.app_data, output_dir, train_paths, val_paths)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Show success message
            message = f"Dataset saved successfully in {format_type.upper()} format"
            if augmented_data and num_augmentations > 0:
                message += f" with {len(augmented_data)} augmented images"
            self.main_window.show_message("Success", f"{message}\nLocation: {output_dir}", QMessageBox.Icon.Information)
            
        except Exception as e:
            self.main_window.show_message("Error", f"Failed to save dataset: {str(e)}", QMessageBox.Icon.Critical)
            import traceback
            traceback.print_exc()
        finally:
            self.main_window.set_ui_busy(False, "Save completed.")
            self.update_button_states()
            # Only show original images in the UI list
            self.main_window.update_image_list(self._get_original_image_paths())

    def add_class(self, class_name: str):
        if class_name in self.app_data.classes:
             self.main_window.show_message("Info", f"Class '{class_name}' already exists.")
             return
        self.app_data.classes.append(class_name)
        self.main_window.update_class_list(self.app_data.classes)
        self.update_button_states()
        
        # Save state after adding a class
        self.state_manager.save_state()

    def remove_class(self, class_name: str):
         if class_name not in self.app_data.classes: return
         class_id_to_remove = self.app_data.classes.index(class_name)
         self.app_data.classes.pop(class_id_to_remove)
         # Handle annotations using the removed class (e.g., set to -1 or prompt user)
         self.data_handler.remap_class_id(class_id_to_remove, -1) # Remap to -1 (invalid)
         self.main_window.update_class_list(self.app_data.classes)
         self.update_button_states()
         # Force redraw of current image if annotations might have changed
         if self.current_image_path:
             self.load_image_and_annotations(self.current_image_path)
             
         # Save state after removing a class
         self.state_manager.save_state()

    def assign_class_to_selected_box(self, class_index: int):
        if self.current_image_path and self.selected_box_canvas_index != -1:
            if 0 <= class_index < len(self.app_data.classes):
                 # Update the specific box in the data model
                 boxes = self.app_data.images[self.current_image_path].boxes
                 if 0 <= self.selected_box_canvas_index < len(boxes):
                     boxes[self.selected_box_canvas_index].class_id = class_index
                     # Tell canvas to redraw
                     self.main_window.get_image_canvas().set_annotations(boxes, self.app_data.classes)
                     
                     # Save state after changing a box's class
                     self.state_manager.save_state()

    # --- Methods Triggered by ImageCanvas ---

    def on_annotations_updated(self):
        """Called when canvas signals a change (move/resize)."""
        # Data is already updated in the canvas's internal list which points
        # to the same BoundingBox objects managed by DataHandler/AppData.
        # May need to mark data as 'dirty' for saving state later.
        self.update_button_states() # Save button might become enabled
        
        # Save state after annotations are updated
        self.state_manager.save_state()

    def on_box_selected_in_canvas(self, box_index: int):
        """Called when canvas signals a box selection change."""
        self.selected_box_canvas_index = box_index
        # Maybe update class list selection to match the selected box's class?
        # Or keep them independent? Let's keep independent for now.

    def on_new_box_drawn(self, pixel_rect: QRect):
        """Called when canvas signals a new box was drawn by user."""
        if not self.current_image_path or not self.app_data.images[self.current_image_path].width:
             print("Error: Cannot add box, image data not ready.")
             return

        img_w = self.app_data.images[self.current_image_path].width
        img_h = self.app_data.images[self.current_image_path].height

        bbox_norm = utils.pixel_to_normalized(
            (pixel_rect.left(), pixel_rect.top(), pixel_rect.right(), pixel_rect.bottom()),
            img_w, img_h
        )
        if bbox_norm:
             # Create new BoundingBox, assign default class (-1 or 0?)
             new_box = BoundingBox(class_id=-1, bbox_norm=bbox_norm, bbox_pixels=pixel_rect.getRect())
             self.app_data.images[self.current_image_path].boxes.append(new_box)
             # Update canvas immediately
             canvas = self.main_window.get_image_canvas()
             canvas.set_annotations(self.app_data.images[self.current_image_path].boxes, self.app_data.classes)
             # Select the newly drawn box
             new_index = len(self.app_data.images[self.current_image_path].boxes) - 1
             canvas.selected_box_idx = new_index
             self.selected_box_canvas_index = new_index
             canvas.update() # Ensure redraw with selection
             self.update_button_states()
             
             # Save state after drawing a new box
             self.state_manager.save_state()

    def on_delete_box_requested(self, box_index: int):
        """Called when canvas signals a delete request (via context menu or shortcut)."""
        if self.current_image_path and 0 <= box_index < len(self.app_data.images[self.current_image_path].boxes):
            del self.app_data.images[self.current_image_path].boxes[box_index]
            # Update canvas
            canvas = self.main_window.get_image_canvas()
            canvas.selected_box_idx = -1 # Deselect after delete
            self.selected_box_canvas_index = -1
            canvas.set_annotations(self.app_data.images[self.current_image_path].boxes, self.app_data.classes)
            self.update_button_states()
            
            # Save state after deleting a box
            self.state_manager.save_state()

    # --- Slots for Worker Signals ---

    def _handle_detection_result(self, result_tuple):
        """Update data model with detection results from worker."""
        image_path, detected_boxes = result_tuple  # Unpack the tuple from the signal
        if image_path in self.app_data.images:
             self.app_data.images[image_path].boxes = detected_boxes
             self.app_data.images[image_path].processed = True
             # If this is the currently viewed image, update the canvas
             if image_path == self.current_image_path:
                 self.load_image_and_annotations(image_path)
                 
             # Save state after processing
             self.state_manager.save_state()

    def _on_detection_finished(self):
        self.main_window.set_ui_busy(False, "Detection finished.")
        self.update_button_states()

    def _handle_augmentation_result(self, augmented_data):
        """Update app data with augmentation results."""
        self.data_handler.add_augmented_data(augmented_data)
        # No need to call update_image_list directly as it's handled in _on_augmentation_finished
        # and will filter out augmented images

    def _on_augmentation_finished(self):
         self.main_window.set_ui_busy(False, "Augmentation finished.")
         self.update_button_states()
         # Only show original images in the UI list, not augmented versions
         self.main_window.update_image_list(self._get_original_image_paths()) # Refresh list after augment

    def _on_save_finished(self, message: str):
         self.main_window.set_ui_busy(False, message) # Show success/completion message
         self.update_button_states()

    def _handle_worker_error(self, error_info):
        """Show error message when a worker thread fails."""
        # error_info could be a tuple (exception_type, exception_value, traceback_str) or just a string
        error_message = f"Background task failed:\n{error_info}"
        print(error_message) # Log detailed error
        self.main_window.show_message("Error", "A background task encountered an error. Check console/logs.", QMessageBox.Icon.Critical)
        self.main_window.set_ui_busy(False, "Error occurred.") # Reset UI
        self.update_button_states()

    def _update_worker_progress(self, value: int, total: int):
        """Update the main progress bar."""
        self.main_window.update_progress(value, total)

    # --- State Checking for UI ---

    def update_button_states(self):
        """Central method to enable/disable buttons based on app state."""
        can_process = bool(self.app_data.model_path) and bool(self.current_image_path)
        can_save = self.data_handler.has_annotations()
        # can_augment = self.data_handler.has_annotations() # Can augment if anything is annotated

        self.main_window.process_button.setEnabled(can_process)
        self.main_window.save_button.setEnabled(can_save)
        # self.main_window.augment_button.setEnabled(can_augment)  # This button no longer exists

    def is_ready_to_process(self) -> bool:
        return bool(self.app_data.model_path) and bool(self.current_image_path)

    def is_ready_to_augment(self) -> bool:
        return self.data_handler.has_annotations()
        
    def is_ready_to_save(self) -> bool:
        return self.data_handler.has_annotations()

    def _on_save_state_requested(self):
        """Handle manual save state request"""
        self.state_manager.save_state()
        self.main_window.show_auto_save_indicator()
        self.main_window.status_bar.showMessage("Application state saved manually", 3000)
        
    def _on_clear_state_requested(self):
        """Handle clear state request"""
        # Clear the saved state files
        self.state_manager.clear_state()
        
        # Reset the app data
        self.app_data.images.clear()
        self.app_data.classes.clear()
        self.app_data.model_path = None
        
        # Update UI
        self.main_window.update_image_list([])
        self.main_window.update_class_list([])
        self.main_window.set_model_label(None)
        
        # Clear the canvas
        canvas = self.main_window.get_image_canvas()
        if canvas:
            canvas.clear()
            canvas.update()
            
        self.current_image_path = None
        self.selected_box_canvas_index = -1
        
        # Update button states
        self.update_button_states()
        
        # Show confirmation
        self.main_window.status_bar.showMessage("Application state reset to defaults", 3000)

    def _on_delete_image_requested(self, image_path: str):
        # Implement the logic to delete the image from the app_data and update the UI
        if image_path in self.app_data.images:
            # Check if the image to be deleted is the current image
            is_current_image = (image_path == self.current_image_path)
            
            # Delete the image
            del self.app_data.images[image_path]
            
            # Update UI - only show original images
            self.main_window.update_image_list(self._get_original_image_paths())
            
            # If the current image was deleted, clear canvas and reset current path
            if is_current_image:
                self.current_image_path = None
                self.selected_box_canvas_index = -1
                self.main_window.get_image_canvas().clear()
            
            self.update_button_states()
            
            # Show status message
            image_name = os.path.basename(image_path)
            self.main_window.status_bar.showMessage(f"Deleted image: {image_name}", 3000)
            
            # Save state after deleting an image
            self.state_manager.save_state()
        else:
            print(f"Error: {image_path} not found in app data.")

    def _on_clear_images_requested(self):
        # Implement the logic to clear all images from the app_data and update the UI
        image_count = len(self.app_data.images)
        if image_count > 0:
            self.app_data.images.clear()
            self.main_window.update_image_list([])
            
            # Clear current image state
            self.current_image_path = None
            self.selected_box_canvas_index = -1
            self.main_window.get_image_canvas().clear()
            
            self.update_button_states()
            
            # Show status message
            self.main_window.status_bar.showMessage(f"Cleared {image_count} images", 3000)
            
            # Save state after clearing images
            self.state_manager.save_state()
        else:
            self.main_window.status_bar.showMessage("No images to clear", 3000)

    def _on_import_classes_requested(self, class_names: list):
        """Handle importing class names from a text file"""
        if not class_names:
            return
            
        # Keep track of what was imported
        imported_count = 0
        skipped_count = 0
        imported_classes = []
        skipped_classes = []
        
        # Process each class name
        for class_name in class_names:
            # Skip empty names or whitespace-only strings
            if not class_name.strip():
                continue
                
            # Skip duplicates
            if class_name in self.app_data.classes:
                skipped_classes.append(class_name)
                skipped_count += 1
                continue
                
            # Add valid class name
            self.app_data.classes.append(class_name)
            imported_classes.append(class_name)
            imported_count += 1
        
        # Update UI
        self.main_window.update_class_list(self.app_data.classes)
        
        # Update canvas if open
        if self.current_image_path and self.current_image_path in self.app_data.images:
            canvas = self.main_window.get_image_canvas()
            canvas.class_names = self.app_data.classes
            canvas.update()  # Refresh the canvas
        
        self.update_button_states()
        
        # Save state after importing classes
        self.state_manager.save_state()
        
        # Show detailed report if anything was processed
        if imported_count > 0 or skipped_count > 0:
            # Create detailed report
            report = f"Import Summary:\n\n"
            
            if imported_count > 0:
                report += f"Successfully imported {imported_count} classes:\n"
                report += "- " + "\n- ".join(imported_classes) + "\n\n"
            
            if skipped_count > 0:
                report += f"Skipped {skipped_count} duplicate classes:\n"
                report += "- " + "\n- ".join(skipped_classes)
            
            # Show in a message box for more detailed view
            self.main_window.show_message(
                "Class Import Report", 
                report, 
                QMessageBox.Icon.Information
            )
        
        # Show brief message in status bar
        if imported_count > 0:
            if skipped_count > 0:
                self.main_window.status_bar.showMessage(
                    f"Imported {imported_count} classes, skipped {skipped_count} duplicates", 
                    3000
                )
            else:
                self.main_window.status_bar.showMessage(
                    f"Successfully imported {imported_count} classes", 
                    3000
                )
        else:
            self.main_window.status_bar.showMessage(
                "No new classes imported (all were duplicates or empty)", 
                3000
            )

    def set_augmentation_settings(self, settings: dict):
        """Apply augmentation settings to the image augmenter.
        
        Args:
            settings: Dictionary containing augmentation settings
        """
        try:
            # Create a new configuration object
            config = self.image_augmenter.config
            
            # Update geometric transform settings
            if "geometric" in settings:
                geo_settings = settings["geometric"]
                config.geometric_transforms_prob = geo_settings.get("probability", 0.5)
                config.hflip_prob = geo_settings.get("hflip_prob", 0.5)
                config.vflip_prob = geo_settings.get("vflip_prob", 0.5)
                config.rotate_prob = geo_settings.get("rotate_prob", 0.3)
                config.rotate_limit = geo_settings.get("rotate_limit", 30)
            
            # Update color transform settings
            if "color" in settings:
                color_settings = settings["color"]
                config.color_transforms_prob = color_settings.get("probability", 0.5)
                config.brightness_contrast_prob = color_settings.get("brightness_contrast_prob", 0.5)
                config.hue_saturation_prob = color_settings.get("hue_saturation_prob", 0.3)
                config.rgb_shift_prob = color_settings.get("rgb_shift_prob", 0.3)
            
            # Update weather transform settings
            if "weather" in settings:
                weather_settings = settings["weather"]
                config.weather_transforms_prob = weather_settings.get("probability", 0.3)
            
            # Update noise transform settings
            if "noise" in settings:
                noise_settings = settings["noise"]
                config.noise_transforms_prob = noise_settings.get("probability", 0.3)
                config.gaussian_noise_prob = noise_settings.get("gaussian_noise_prob", 0.3)
            
            # Update blur transform settings
            if "blur" in settings:
                blur_settings = settings["blur"]
                config.blur_transforms_prob = blur_settings.get("probability", 0.3)
                config.blur_prob = blur_settings.get("blur_prob", 0.3)
            
            # Store enabled/disabled states for transform categories
            self.image_augmenter.enabled_transforms = {
                "geometric": settings.get("geometric", {}).get("enabled", True),
                "color": settings.get("color", {}).get("enabled", True),
                "weather": settings.get("weather", {}).get("enabled", True),
                "noise": settings.get("noise", {}).get("enabled", True),
                "blur": settings.get("blur", {}).get("enabled", True)
            }
            
            # Log the changes
            print("Augmentation settings updated successfully")
            
        except Exception as e:
            print(f"Error updating augmentation settings: {str(e)}")
            self.main_window.show_message("Error", f"Failed to update augmentation settings: {str(e)}", QMessageBox.Icon.Critical)

    def _get_original_image_paths(self):
        """Returns list of original (non-augmented) image paths that exist on disk."""
        return [path for path, annot in self.app_data.images.items() 
                if annot.augmented_from is None and os.path.exists(path)]
    
    def on_class_assignment_requested(self, box_index: int, class_id: int):
        """Handle class assignment request from drag and drop."""
        if self.current_image_path and 0 <= box_index < len(self.app_data.images[self.current_image_path].boxes):
            # Assign the class to the box
            self.app_data.images[self.current_image_path].boxes[box_index].class_id = class_id
            
            # Update the canvas
            canvas = self.main_window.get_image_canvas()
            canvas.set_annotations(self.app_data.images[self.current_image_path].boxes, self.app_data.classes)
            self.update_button_states()
            
            # Save state after assigning class
            self.state_manager.save_state()