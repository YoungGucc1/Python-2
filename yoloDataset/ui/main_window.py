"""
MainWindow module - Main window for the YOLO Dataset Creator application
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QPushButton, QListWidget, QListWidgetItem, QLabel, QProgressBar, QSpinBox,
    QLineEdit, QStatusBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

import os

from .image_canvas import ImageCanvas

class MainWindow(QMainWindow):
    """
    Main window for the YOLO Dataset Creator application
    """
    
    # Signals
    addImagesClicked = Signal(list)  # Emits list of file paths
    selectModelClicked = Signal(str)  # Emits model file path
    processImageClicked = Signal()    # Emits when process button clicked
    saveDatasetClicked = Signal(str)  # Emits output directory
    augmentDatasetClicked = Signal(int)  # Emits number of augmentations
    addClassClicked = Signal(str)     # Emits class name
    removeClassClicked = Signal(str)  # Emits class name
    imageSelected = Signal(str)       # Emits image path
    classSelected = Signal(str)       # Emits class name
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        self.setWindowTitle("YOLO Dataset Creator")
        self.resize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create and add controls panel
        controls_panel = self._create_controls_panel()
        main_layout.addWidget(controls_panel)
        
        # Create splitter for the main areas
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create and add image files panel
        images_panel = self._create_images_panel()
        main_splitter.addWidget(images_panel)
        
        # Create and add image annotation panel
        annotation_panel = self._create_annotation_panel()
        main_splitter.addWidget(annotation_panel)
        
        # Create and add classes panel
        classes_panel = self._create_classes_panel()
        main_splitter.addWidget(classes_panel)
        
        # Set splitter sizes (roughly 1:3:1 ratio)
        main_splitter.setSizes([200, 600, 200])
        
        # Create and set status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect internal signals/slots
        self._connect_signals()
    
    def _create_controls_panel(self):
        """
        Create the controls panel (top area)
        
        Returns:
            QWidget: Controls panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Row 1: Add Images, Select Model
        row1 = QHBoxLayout()
        
        self.add_images_btn = QPushButton("Add Images...")
        row1.addWidget(self.add_images_btn)
        
        self.select_model_btn = QPushButton("Select Model...")
        row1.addWidget(self.select_model_btn)
        
        self.model_label = QLabel("Selected Model: None")
        row1.addWidget(self.model_label)
        
        row1.addStretch()
        
        layout.addLayout(row1)
        
        # Row 2: Process Image, Progress
        row2 = QHBoxLayout()
        
        self.process_btn = QPushButton("Process Selected Image")
        self.process_btn.setEnabled(False)  # Disabled until an image and model are selected
        row2.addWidget(self.process_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        row2.addWidget(self.progress_bar)
        
        layout.addLayout(row2)
        
        # Row 3: Augment, Save
        row3 = QHBoxLayout()
        
        self.augment_btn = QPushButton("Augment Dataset")
        self.augment_btn.setEnabled(False)  # Disabled until images are processed
        row3.addWidget(self.augment_btn)
        
        row3.addWidget(QLabel("Augmentations per image:"))
        
        self.augmentations_spin = QSpinBox()
        self.augmentations_spin.setRange(1, 20)
        self.augmentations_spin.setValue(5)
        row3.addWidget(self.augmentations_spin)
        
        row3.addStretch()
        
        self.save_btn = QPushButton("Save Dataset...")
        self.save_btn.setEnabled(False)  # Disabled until images are processed
        row3.addWidget(self.save_btn)
        
        layout.addLayout(row3)
        
        return panel
    
    def _create_images_panel(self):
        """
        Create the images panel (left area)
        
        Returns:
            QWidget: Images panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("Image Files"))
        
        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.images_list)
        
        self.images_status_label = QLabel("Loaded: 0 images")
        layout.addWidget(self.images_status_label)
        
        return panel
    
    def _create_annotation_panel(self):
        """
        Create the image annotation panel (center area)
        
        Returns:
            QWidget: Annotation panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("Image Annotation Area"))
        
        # Create image canvas
        self.image_canvas = ImageCanvas()
        layout.addWidget(self.image_canvas)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom +")
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom -")
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.fit_btn = QPushButton("Fit")
        zoom_layout.addWidget(self.fit_btn)
        
        zoom_layout.addStretch()
        
        layout.addLayout(zoom_layout)
        
        # Instructions
        instructions = QLabel("Click on a box to select it, then click a class to assign it.")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        return panel
    
    def _create_classes_panel(self):
        """
        Create the classes panel (right area)
        
        Returns:
            QWidget: Classes panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("Classes"))
        
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.classes_list)
        
        # Add new class controls
        layout.addWidget(QLabel("Add New Class:"))
        
        self.new_class_edit = QLineEdit()
        layout.addWidget(self.new_class_edit)
        
        add_class_layout = QHBoxLayout()
        
        self.add_class_btn = QPushButton("Add Class")
        add_class_layout.addWidget(self.add_class_btn)
        
        self.remove_class_btn = QPushButton("Remove Selected")
        self.remove_class_btn.setEnabled(False)  # Disabled until a class is selected
        add_class_layout.addWidget(self.remove_class_btn)
        
        layout.addLayout(add_class_layout)
        
        # Instructions
        instructions = QLabel("Select a box, then click a class to assign it.")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        return panel
    
    def _connect_signals(self):
        """Connect internal widget signals to slots"""
        # Button clicks
        self.add_images_btn.clicked.connect(self._on_add_images_clicked)
        self.select_model_btn.clicked.connect(self._on_select_model_clicked)
        self.process_btn.clicked.connect(self.processImageClicked)
        self.save_btn.clicked.connect(self._on_save_dataset_clicked)
        self.augment_btn.clicked.connect(self._on_augment_dataset_clicked)
        self.add_class_btn.clicked.connect(self._on_add_class_clicked)
        self.remove_class_btn.clicked.connect(self._on_remove_class_clicked)
        
        # Image canvas controls
        self.zoom_in_btn.clicked.connect(self.image_canvas.zoom_in)
        self.zoom_out_btn.clicked.connect(self.image_canvas.zoom_out)
        self.fit_btn.clicked.connect(self.image_canvas.fit_to_view)
        
        # List selections
        self.images_list.currentItemChanged.connect(self._on_image_selected)
        self.classes_list.currentItemChanged.connect(self._on_class_selected)
        
        # New class input
        self.new_class_edit.returnPressed.connect(self._on_add_class_clicked)
    
    def _on_add_images_clicked(self):
        """Handle Add Images button click"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        
        if file_paths:
            self.addImagesClicked.emit(file_paths)
    
    def _on_select_model_clicked(self):
        """Handle Select Model button click"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "YOLO Models (*.pt *.onnx)"
        )
        
        if file_path:
            self.selectModelClicked.emit(file_path)
    
    def _on_save_dataset_clicked(self):
        """Handle Save Dataset button click"""
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if output_dir:
            self.saveDatasetClicked.emit(output_dir)
    
    def _on_augment_dataset_clicked(self):
        """Handle Augment Dataset button click"""
        num_augmentations = self.augmentations_spin.value()
        self.augmentDatasetClicked.emit(num_augmentations)
    
    def _on_add_class_clicked(self):
        """Handle Add Class button click"""
        class_name = self.new_class_edit.text().strip()
        if class_name:
            self.addClassClicked.emit(class_name)
            self.new_class_edit.clear()
    
    def _on_remove_class_clicked(self):
        """Handle Remove Class button click"""
        selected_items = self.classes_list.selectedItems()
        if selected_items:
            class_name = selected_items[0].text()
            self.removeClassClicked.emit(class_name)
    
    def _on_image_selected(self, current, previous):
        """
        Handle image selection in the list
        
        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        if current:
            # Get the image path from the item's data
            image_path = current.data(Qt.UserRole)
            self.imageSelected.emit(image_path)
    
    def _on_class_selected(self, current, previous):
        """
        Handle class selection in the list
        
        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        if current:
            class_name = current.text()
            self.classSelected.emit(class_name)
            self.remove_class_btn.setEnabled(True)
        else:
            self.remove_class_btn.setEnabled(False)
    
    def add_images_to_list(self, image_paths):
        """
        Add images to the images list
        
        Args:
            image_paths (list): List of image paths to add
        """
        for path in image_paths:
            item_text = os.path.basename(path)
            items = self.images_list.findItems(item_text, Qt.MatchExactly)
            
            if not items:  # Only add if not already in the list
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, path)
                self.images_list.addItem(item)
        
        # Update status label
        self.images_status_label.setText(f"Loaded: {self.images_list.count()} images")
        
        # Enable the process button if we have images and a model
        self._update_buttons_state()
        
        # Select the first item if none selected
        if self.images_list.count() > 0 and not self.images_list.selectedItems():
            self.images_list.setCurrentRow(0)
    
    def set_model_path(self, model_path):
        """
        Set the model path label
        
        Args:
            model_path (str): Path to the model
        """
        model_name = os.path.basename(model_path)
        self.model_label.setText(f"Selected Model: {model_name}")
        
        # Enable the process button if we have images and a model
        self._update_buttons_state()
    
    def add_class_to_list(self, class_name):
        """
        Add a class to the classes list
        
        Args:
            class_name (str): Name of the class to add
        """
        # Check if the class already exists
        items = self.classes_list.findItems(class_name, Qt.MatchExactly)
        if not items:
            self.classes_list.addItem(class_name)
    
    def remove_class_from_list(self, class_name):
        """
        Remove a class from the classes list
        
        Args:
            class_name (str): Name of the class to remove
        """
        items = self.classes_list.findItems(class_name, Qt.MatchExactly)
        for item in items:
            self.classes_list.takeItem(self.classes_list.row(item))
    
    def get_selected_image_path(self):
        """
        Get the path of the currently selected image
        
        Returns:
            str: Path to the selected image, or None if no image is selected
        """
        selected_items = self.images_list.selectedItems()
        if selected_items:
            return selected_items[0].data(Qt.UserRole)
        return None
    
    def get_selected_class_name(self):
        """
        Get the name of the currently selected class
        
        Returns:
            str: Name of the selected class, or None if no class is selected
        """
        selected_items = self.classes_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None
    
    def get_class_names(self):
        """
        Get all class names in the list
        
        Returns:
            list: List of all class names
        """
        return [self.classes_list.item(i).text() for i in range(self.classes_list.count())]
    
    def display_image(self, image_path):
        """
        Display an image in the canvas
        
        Args:
            image_path (str): Path to the image to display
        """
        self.image_canvas.set_image(image_path)
    
    def update_annotations(self, annotations):
        """
        Update the annotations displayed in the canvas
        
        Args:
            annotations (list): List of annotations
        """
        self.image_canvas.set_annotations(annotations)
    
    def update_class_names(self, class_names):
        """
        Update the class names in the canvas
        
        Args:
            class_names (list): List of class names
        """
        self.image_canvas.set_class_names(class_names)
    
    def set_progress(self, value, max_value=100):
        """
        Set the progress bar value
        
        Args:
            value (int): Current progress value
            max_value (int, optional): Maximum progress value. Defaults to 100.
        """
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setValue(value)
    
    def set_status(self, message):
        """
        Set the status bar message
        
        Args:
            message (str): Status message
        """
        self.status_bar.showMessage(message)
    
    def show_message(self, title, message, icon=QMessageBox.Information):
        """
        Show a message box
        
        Args:
            title (str): Message box title
            message (str): Message text
            icon (QMessageBox.Icon, optional): Message box icon. Defaults to QMessageBox.Information.
        """
        QMessageBox.information(self, title, message, icon)
    
    def _update_buttons_state(self):
        """Update the enabled state of buttons based on current state"""
        has_images = self.images_list.count() > 0
        has_model = self.model_label.text() != "Selected Model: None"
        has_classes = self.classes_list.count() > 0
        
        # Process button requires images and model
        self.process_btn.setEnabled(has_images and has_model)
        
        # Augment and Save buttons require images that have been processed
        # This will be updated separately by the controller
    
    def enable_augment_button(self, enable=True):
        """
        Enable or disable the Augment Dataset button
        
        Args:
            enable (bool, optional): Whether to enable the button. Defaults to True.
        """
        self.augment_btn.setEnabled(enable)
    
    def enable_save_button(self, enable=True):
        """
        Enable or disable the Save Dataset button
        
        Args:
            enable (bool, optional): Whether to enable the button. Defaults to True.
        """
        self.save_btn.setEnabled(enable) 