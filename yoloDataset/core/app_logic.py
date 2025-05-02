"""
AppLogic module - Connects UI events to backend logic
"""

import os
import cv2
from PySide6.QtCore import QObject, Signal, QThread, Slot, QMutex
import numpy as np

# Use absolute imports
from yolo_dataset_creator.core.data_handler import DataHandler
from yolo_dataset_creator.core.yolo_processor import YoloProcessor
from yolo_dataset_creator.core.image_augmenter import ImageAugmenter

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    progress = Signal(int, int, str)  # current, total, status message
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

class DetectionWorker(QThread):
    """Worker thread for running object detection"""
    
    def __init__(self, yolo_processor, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """Initialize the detection worker"""
        super().__init__()
        self.signals = WorkerSignals()
        self.yolo_processor = yolo_processor
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def run(self):
        """Run detection on the image"""
        try:
            self.signals.progress.emit(0, 1, f"Detecting objects in {os.path.basename(self.image_path)}...")
            
            # Run detection
            detections = self.yolo_processor.detect(
                self.image_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            if detections is None:
                self.signals.error.emit(f"Detection failed for {os.path.basename(self.image_path)}")
                return
                
            self.signals.progress.emit(1, 1, f"Detection completed: {len(detections)} objects found")
            
            # Emit the result
            self.signals.result.emit(detections)
            
        except Exception as e:
            self.signals.error.emit(f"Error during detection: {str(e)}")
        finally:
            self.signals.finished.emit()

class AugmentationWorker(QThread):
    """Worker thread for augmenting images"""
    
    def __init__(self, image_augmenter, data_handler, num_augmentations):
        """Initialize the augmentation worker"""
        super().__init__()
        self.signals = WorkerSignals()
        self.image_augmenter = image_augmenter
        self.data_handler = data_handler
        self.num_augmentations = num_augmentations
        self.mutex = QMutex()
        
    def run(self):
        """Run augmentation on all annotated images"""
        try:
            # Get list of images with annotations
            image_paths = [path for path in self.data_handler.image_paths 
                          if path in self.data_handler.annotations and self.data_handler.annotations[path]]
            
            if not image_paths:
                self.signals.error.emit("No annotated images to augment")
                return
                
            total_steps = len(image_paths)
            
            for i, image_path in enumerate(image_paths):
                self.signals.progress.emit(i, total_steps, f"Augmenting image {i+1}/{total_steps}...")
                
                # Get annotations for this image
                annotations = self.data_handler.annotations[image_path]
                
                if not annotations:
                    continue
                    
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                # Generate augmentations
                augmented_data = self.image_augmenter.augment_image(img, annotations, self.num_augmentations)
                
                # Store augmented images
                with QMutex():
                    self.data_handler.add_augmented_images(image_path, augmented_data)
                
            self.signals.progress.emit(total_steps, total_steps, f"Augmentation completed: {self.num_augmentations} variations for {len(image_paths)} images")
            self.signals.result.emit(True)
            
        except Exception as e:
            self.signals.error.emit(f"Error during augmentation: {str(e)}")
        finally:
            self.signals.finished.emit()

class SaveDatasetWorker(QThread):
    """Worker thread for saving the dataset"""
    
    def __init__(self, data_handler, output_dir, train_split=0.8):
        """Initialize the save dataset worker"""
        super().__init__()
        self.signals = WorkerSignals()
        self.data_handler = data_handler
        self.output_dir = output_dir
        self.train_split = train_split
        
    def run(self):
        """Save the dataset"""
        try:
            # Define a progress callback for data_handler
            def progress_callback(current, total, message):
                self.signals.progress.emit(current, total, message)
            
            # Save the dataset
            success = self.data_handler.save_yolo_dataset(
                self.output_dir,
                train_split=self.train_split,
                progress_callback=progress_callback
            )
            
            if not success:
                self.signals.error.emit("Failed to save dataset")
                return
                
            self.signals.result.emit(True)
            
        except Exception as e:
            self.signals.error.emit(f"Error saving dataset: {str(e)}")
        finally:
            self.signals.finished.emit()

class AppLogic(QObject):
    """
    Application logic controller - connects UI events to backend logic
    """
    
    # Signals to update UI
    statusChanged = Signal(str)
    progressChanged = Signal(int, int)
    imageDisplayed = Signal(str)
    annotationsUpdated = Signal(list)
    classNamesUpdated = Signal(list)
    augmentButtonEnabled = Signal(bool)
    saveButtonEnabled = Signal(bool)
    
    def __init__(self, main_window):
        """
        Initialize the application logic
        
        Args:
            main_window: MainWindow instance
        """
        super().__init__()
        
        self.main_window = main_window
        
        # Initialize core components
        self.data_handler = DataHandler()
        self.yolo_processor = YoloProcessor()
        self.image_augmenter = ImageAugmenter()
        
        # Current state
        self.current_image_path = None
        self.detection_worker = None
        self.augmentation_worker = None
        self.save_worker = None
        
        # Connect signals from UI
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect signals and slots between UI and logic"""
        # Connect UI signals to logic slots
        self.main_window.addImagesClicked.connect(self._on_add_images)
        self.main_window.selectModelClicked.connect(self._on_select_model)
        self.main_window.processImageClicked.connect(self._on_process_image)
        self.main_window.saveDatasetClicked.connect(self._on_save_dataset)
        self.main_window.augmentDatasetClicked.connect(self._on_augment_dataset)
        self.main_window.addClassClicked.connect(self._on_add_class)
        self.main_window.removeClassClicked.connect(self._on_remove_class)
        self.main_window.imageSelected.connect(self._on_image_selected)
        self.main_window.classSelected.connect(self._on_class_selected)
        
        # Connect the ImageCanvas boxSelected signal
        self.main_window.image_canvas.boxSelected.connect(self._on_box_selected)
        
        # Connect logic signals to UI slots
        self.statusChanged.connect(self.main_window.set_status)
        self.progressChanged.connect(lambda current, total: self.main_window.set_progress(current, total))
        self.imageDisplayed.connect(self.main_window.display_image)
        self.annotationsUpdated.connect(self.main_window.update_annotations)
        self.classNamesUpdated.connect(self.main_window.update_class_names)
        self.augmentButtonEnabled.connect(self.main_window.enable_augment_button)
        self.saveButtonEnabled.connect(self.main_window.enable_save_button)
    
    def _on_add_images(self, file_paths):
        """
        Handle adding images
        
        Args:
            file_paths (list): List of image file paths
        """
        # Add images to the data handler
        added_count = self.data_handler.add_images(file_paths)
        
        # Update UI
        self.main_window.add_images_to_list(file_paths)
        self.statusChanged.emit(f"Added {added_count} images")
        
        # Update button states
        self._update_ui_state()
    
    def _on_select_model(self, model_path):
        """
        Handle selecting a model
        
        Args:
            model_path (str): Path to the model file
        """
        # Set model path in data handler
        if self.data_handler.set_model_path(model_path):
            # Try to load the model
            if self.yolo_processor.load_model(model_path):
                # Update UI
                self.main_window.set_model_path(model_path)
                self.statusChanged.emit(f"Loaded model: {os.path.basename(model_path)}")
            else:
                self.statusChanged.emit(f"Failed to load model: {os.path.basename(model_path)}")
                self.main_window.show_message(
                    "Model Load Error",
                    f"Failed to load the model: {os.path.basename(model_path)}",
                    icon=self.main_window.QMessageBox.Warning
                )
        else:
            self.statusChanged.emit(f"Invalid model path: {model_path}")
    
    def _on_process_image(self):
        """Handle processing the selected image"""
        # Get the selected image path
        image_path = self.main_window.get_selected_image_path()
        if not image_path:
            self.statusChanged.emit("No image selected")
            return
            
        # Check if model is loaded
        if self.yolo_processor.model is None:
            self.statusChanged.emit("No model loaded")
            return
            
        # Create and run detection thread
        self.detection_worker = DetectionWorker(
            self.yolo_processor,
            image_path,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        # Connect worker signals
        self.detection_worker.signals.progress.connect(
            lambda current, total, message: self._handle_worker_progress(current, total, message)
        )
        self.detection_worker.signals.result.connect(
            lambda detections: self._handle_detection_results(detections, image_path)
        )
        self.detection_worker.signals.error.connect(
            lambda error_msg: self._handle_worker_error(error_msg)
        )
        self.detection_worker.signals.finished.connect(
            lambda: self._handle_worker_finished()
        )
        
        # Start the worker
        self.detection_worker.start()
        
        # Update UI
        self.statusChanged.emit(f"Processing image: {os.path.basename(image_path)}...")
    
    def _on_save_dataset(self, output_dir):
        """
        Handle saving the dataset
        
        Args:
            output_dir (str): Output directory for the dataset
        """
        # Validate output directory
        if not output_dir:
            self.statusChanged.emit("No output directory selected")
            return
            
        # Check if we have any annotated images
        has_annotations = any(bool(annotations) for annotations in self.data_handler.annotations.values())
        if not has_annotations:
            self.statusChanged.emit("No annotated images to save")
            self.main_window.show_message(
                "No Data to Save",
                "There are no annotated images to save. Please process at least one image first.",
                icon=self.main_window.QMessageBox.Warning
            )
            return
            
        # Check if we have classes defined
        if not self.data_handler.class_names:
            self.statusChanged.emit("No classes defined")
            self.main_window.show_message(
                "No Classes Defined",
                "Please define at least one class before saving the dataset.",
                icon=self.main_window.QMessageBox.Warning
            )
            return
            
        # Create and run save dataset thread
        self.save_worker = SaveDatasetWorker(
            self.data_handler,
            output_dir,
            train_split=0.8
        )
        
        # Connect worker signals
        self.save_worker.signals.progress.connect(
            lambda current, total, message: self._handle_worker_progress(current, total, message)
        )
        self.save_worker.signals.result.connect(
            lambda _: self._handle_save_results(output_dir)
        )
        self.save_worker.signals.error.connect(
            lambda error_msg: self._handle_worker_error(error_msg)
        )
        self.save_worker.signals.finished.connect(
            lambda: self._handle_worker_finished()
        )
        
        # Start the worker
        self.save_worker.start()
        
        # Update UI
        self.statusChanged.emit(f"Saving dataset to: {output_dir}...")
    
    def _on_augment_dataset(self, num_augmentations):
        """
        Handle augmenting the dataset
        
        Args:
            num_augmentations (int): Number of augmentations to create per image
        """
        # Check if we have any annotated images
        has_annotations = any(bool(annotations) for annotations in self.data_handler.annotations.values())
        if not has_annotations:
            self.statusChanged.emit("No annotated images to augment")
            self.main_window.show_message(
                "No Data to Augment",
                "There are no annotated images to augment. Please process at least one image first.",
                icon=self.main_window.QMessageBox.Warning
            )
            return
            
        # Create and run augmentation thread
        self.augmentation_worker = AugmentationWorker(
            self.image_augmenter,
            self.data_handler,
            num_augmentations
        )
        
        # Connect worker signals
        self.augmentation_worker.signals.progress.connect(
            lambda current, total, message: self._handle_worker_progress(current, total, message)
        )
        self.augmentation_worker.signals.result.connect(
            lambda _: self._handle_augmentation_results()
        )
        self.augmentation_worker.signals.error.connect(
            lambda error_msg: self._handle_worker_error(error_msg)
        )
        self.augmentation_worker.signals.finished.connect(
            lambda: self._handle_worker_finished()
        )
        
        # Start the worker
        self.augmentation_worker.start()
        
        # Update UI
        self.statusChanged.emit(f"Augmenting dataset with {num_augmentations} variations per image...")
    
    def _on_add_class(self, class_name):
        """
        Handle adding a class
        
        Args:
            class_name (str): Name of the class to add
        """
        # Add class to data handler
        if self.data_handler.add_class(class_name):
            # Update UI
            self.main_window.add_class_to_list(class_name)
            self.classNamesUpdated.emit(self.data_handler.class_names)
            self.statusChanged.emit(f"Added class: {class_name}")
        else:
            self.statusChanged.emit(f"Failed to add class: {class_name}")
    
    def _on_remove_class(self, class_name):
        """
        Handle removing a class
        
        Args:
            class_name (str): Name of the class to remove
        """
        # Remove class from data handler
        if self.data_handler.remove_class(class_name):
            # Update UI
            self.main_window.remove_class_from_list(class_name)
            self.classNamesUpdated.emit(self.data_handler.class_names)
            self.statusChanged.emit(f"Removed class: {class_name}")
            
            # Refresh annotations display if current image has annotations
            if self.current_image_path and self.current_image_path in self.data_handler.annotations:
                self.annotationsUpdated.emit(self.data_handler.annotations[self.current_image_path])
        else:
            self.statusChanged.emit(f"Failed to remove class: {class_name}")
    
    def _on_image_selected(self, image_path):
        """
        Handle selecting an image
        
        Args:
            image_path (str): Path to the selected image
        """
        self.current_image_path = image_path
        
        # Display the image
        self.imageDisplayed.emit(image_path)
        
        # Display annotations if any
        if image_path in self.data_handler.annotations:
            self.annotationsUpdated.emit(self.data_handler.annotations[image_path])
        else:
            self.annotationsUpdated.emit([])
            
        self.statusChanged.emit(f"Selected image: {os.path.basename(image_path)}")
    
    def _on_class_selected(self, class_name):
        """
        Handle selecting a class
        
        Args:
            class_name (str): Name of the selected class
        """
        # This is handled in _on_box_selected if a box is selected
        self.statusChanged.emit(f"Selected class: {class_name}")
    
    def _on_box_selected(self, box_index):
        """
        Handle selecting a bounding box
        
        Args:
            box_index (int): Index of the selected box
        """
        if box_index < 0 or not self.current_image_path:
            return
            
        # Get the selected class if one is selected
        class_name = self.main_window.get_selected_class_name()
        if class_name:
            # Get class ID from name
            if class_name in self.data_handler.class_names:
                class_id = self.data_handler.class_names.index(class_name)
                
                # Assign class to box
                if self.data_handler.assign_class_to_box(self.current_image_path, box_index, class_id):
                    # Update UI
                    self.annotationsUpdated.emit(self.data_handler.annotations[self.current_image_path])
                    self.statusChanged.emit(f"Assigned class '{class_name}' to box {box_index+1}")
                    
                    # Enable save button if any boxes have been assigned
                    has_assigned_boxes = any(ann.get('class_id', -1) >= 0 
                                           for annotations in self.data_handler.annotations.values()
                                           for ann in annotations)
                    if has_assigned_boxes:
                        self.saveButtonEnabled.emit(True)
                        self.augmentButtonEnabled.emit(True)
            else:
                self.statusChanged.emit(f"Selected box {box_index+1} (no class assigned)")
        else:
            self.statusChanged.emit(f"Selected box {box_index+1} (no class selected)")
    
    def _handle_detection_results(self, detections, image_path):
        """
        Handle detection results
        
        Args:
            detections (list): List of detected objects
            image_path (str): Path to the image
        """
        # Update annotations in data handler
        self.data_handler.update_annotations(image_path, detections)
        
        # Update UI
        if image_path == self.current_image_path:
            self.annotationsUpdated.emit(self.data_handler.annotations[image_path])
            
        self.statusChanged.emit(f"Detected {len(detections)} objects in {os.path.basename(image_path)}")
        
        # Enable save and augment buttons if we have any annotations
        has_annotations = bool(detections)
        if has_annotations:
            self.saveButtonEnabled.emit(True)
            self.augmentButtonEnabled.emit(True)
    
    def _handle_augmentation_results(self):
        """Handle augmentation results"""
        # Count total augmented images
        total_augmented = sum(len(augmented_list) for augmented_list in self.data_handler.augmented_images.values())
        
        self.statusChanged.emit(f"Augmentation completed: {total_augmented} augmented images created")
        self.main_window.show_message(
            "Augmentation Complete",
            f"Successfully created {total_augmented} augmented images.",
            icon=self.main_window.QMessageBox.Information
        )
    
    def _handle_save_results(self, output_dir):
        """
        Handle save dataset results
        
        Args:
            output_dir (str): Output directory
        """
        dataset_dir = os.path.join(output_dir, 'dataset')
        
        self.statusChanged.emit(f"Dataset saved to: {dataset_dir}")
        self.main_window.show_message(
            "Dataset Saved",
            f"Dataset successfully saved to {dataset_dir}",
            icon=self.main_window.QMessageBox.Information
        )
    
    def _handle_worker_progress(self, current, total, message):
        """
        Handle worker progress updates
        
        Args:
            current (int): Current progress
            total (int): Total steps
            message (str): Progress message
        """
        self.progressChanged.emit(current, total)
        self.statusChanged.emit(message)
    
    def _handle_worker_error(self, error_msg):
        """
        Handle worker errors
        
        Args:
            error_msg (str): Error message
        """
        self.statusChanged.emit(f"Error: {error_msg}")
        self.main_window.show_message(
            "Error",
            error_msg,
            icon=self.main_window.QMessageBox.Warning
        )
    
    def _handle_worker_finished(self):
        """Handle worker finished signal"""
        # Reset progress bar
        self.progressChanged.emit(0, 100)
    
    def _update_ui_state(self):
        """Update UI button states based on application state"""
        # Update class names in the canvas
        self.classNamesUpdated.emit(self.data_handler.class_names)
        
        # Update button states
        has_annotations = any(bool(annotations) for annotations in self.data_handler.annotations.values())
        self.saveButtonEnabled.emit(has_annotations)
        self.augmentButtonEnabled.emit(has_annotations) 