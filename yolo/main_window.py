"""
MainWindow module - Main window for the YOLO Dataset Creator application (PyQt6)
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QLineEdit, QStatusBar, QFileDialog, QMessageBox, QStyleOption, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPainter, QColor, QPalette

# Use local imports due to flat structure
from image_canvas import ImageCanvas

class MainWindow(QMainWindow):
    """
    Main window for the YOLO Dataset Creator application.
    """
    # --- Signals for AppLogic ---
    addImagesClicked = pyqtSignal(list)         # Emits list of file paths
    selectModelClicked = pyqtSignal(str)        # Emits model file path
    processImageClicked = pyqtSignal()          # Emits when process button clicked
    saveDatasetClicked = pyqtSignal(str)        # Emits output directory
    augmentDatasetClicked = pyqtSignal(int)     # Emits number of augmentations
    addClassClicked = pyqtSignal(str)           # Emits class name to add
    removeClassClicked = pyqtSignal(str)        # Emits class name to remove
    imageSelected = pyqtSignal(str)             # Emits image path when selected in list
    classSelected = pyqtSignal(str)             # Emits class name when selected in list
    configChanged = pyqtSignal(dict)            # Emits dict of config values (conf, iou, split)
    drawBoxClicked = pyqtSignal(bool)           # Emits True when Draw Box is toggled on, False off
    deleteSelectedBoxClicked = pyqtSignal()     # Emits when Delete Selected Box is clicked

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset Creator (PyQt6)")
        self.resize(1300, 850) # Slightly larger default size

        self._create_actions() # Optional: For menus/toolbars
        self._create_widgets()
        self._create_layout()
        self._create_connections()
        self._update_buttons_state() # Initial state

        self.status_bar.showMessage("Ready")

    def _create_actions(self):
        # Placeholder for potential menu actions (File, Edit, View, etc.)
        pass

    def _create_widgets(self):
        """Create all the widgets for the main window."""
        # --- Top Controls ---
        self.add_images_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), " Add Images...")
        self.select_model_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), " Select Model...")
        self.model_label = QLabel("Selected Model: None")
        self.model_label.setWordWrap(True)

        self.process_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), " Process Image")
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setPrefix("Conf: ")
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setPrefix("IoU: ")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False) # Status bar shows messages

        self.augment_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), " Augment")
        self.augmentations_spin = QSpinBox()
        self.augmentations_spin.setRange(1, 50) # Increased range
        self.augmentations_spin.setValue(5)
        self.augmentations_spin.setPrefix("Count: ")

        self.save_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), " Save Dataset...")
        self.train_split_spinbox = QDoubleSpinBox()
        self.train_split_spinbox.setRange(0.1, 0.9)
        self.train_split_spinbox.setSingleStep(0.05)
        self.train_split_spinbox.setValue(0.8)
        self.train_split_spinbox.setPrefix("Train Split: ")
        self.train_split_spinbox.setDecimals(2)

        # --- Left Panel (Images) ---
        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.images_status_label = QLabel("Loaded: 0 images")

        # --- Center Panel (Annotation) ---
        self.image_canvas = ImageCanvas()
        self.zoom_in_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), " Zoom +")
        self.zoom_out_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown), " Zoom -")
        self.fit_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), " Fit") # Icon choice is arbitrary
        self.draw_box_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), " Draw Box") # Icon choice arbitrary
        self.draw_box_btn.setCheckable(True)
        self.delete_box_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), " Delete Box")

        # --- Right Panel (Classes) ---
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("Enter new class name...")
        self.add_class_btn = QPushButton("Add Class")
        self.remove_class_btn = QPushButton("Remove Selected")

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _create_layout(self):
        """Create the main layout and add widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls Panel Layout
        controls_panel = QWidget()
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.addWidget(self.add_images_btn)
        controls_layout.addWidget(self.select_model_btn)
        controls_layout.addWidget(self.model_label, 1) # Stretch label
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(self.conf_spinbox)
        controls_layout.addWidget(self.iou_spinbox)
        controls_layout.addWidget(self.progress_bar, 1) # Stretch progress bar
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.augment_btn)
        controls_layout.addWidget(self.augmentations_spin)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.train_split_spinbox)
        main_layout.addWidget(controls_panel)

        # Main Splitter Layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter, 1) # Make splitter take remaining space

        # Images Panel (Left)
        images_panel = QWidget()
        images_layout = QVBoxLayout(images_panel)
        images_layout.addWidget(QLabel("<b>Image Files</b>"))
        images_layout.addWidget(self.images_list, 1)
        images_layout.addWidget(self.images_status_label)
        main_splitter.addWidget(images_panel)

        # Annotation Panel (Center)
        annotation_panel = QWidget()
        annotation_layout = QVBoxLayout(annotation_panel)
        annotation_layout.addWidget(QLabel("<b>Annotation Area</b>"))
        annotation_layout.addWidget(self.image_canvas, 1) # Canvas takes most space

        canvas_controls_layout = QHBoxLayout()
        canvas_controls_layout.addWidget(self.zoom_in_btn)
        canvas_controls_layout.addWidget(self.zoom_out_btn)
        canvas_controls_layout.addWidget(self.fit_btn)
        canvas_controls_layout.addStretch(1)
        canvas_controls_layout.addWidget(self.draw_box_btn)
        canvas_controls_layout.addWidget(self.delete_box_btn)
        annotation_layout.addLayout(canvas_controls_layout)

        instructions = QLabel("Ctrl+Scroll: Zoom | Middle Mouse/Ctrl+Drag: Pan | Click Box: Select | Click Class: Assign/Select")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        annotation_layout.addWidget(instructions)
        main_splitter.addWidget(annotation_panel)

        # Classes Panel (Right)
        classes_panel = QWidget()
        classes_layout = QVBoxLayout(classes_panel)
        classes_layout.addWidget(QLabel("<b>Classes</b>"))
        classes_layout.addWidget(self.classes_list, 1)
        classes_layout.addWidget(QLabel("Add New Class:"))
        classes_layout.addWidget(self.new_class_edit)
        add_remove_layout = QHBoxLayout()
        add_remove_layout.addWidget(self.add_class_btn)
        add_remove_layout.addWidget(self.remove_class_btn)
        classes_layout.addLayout(add_remove_layout)
        main_splitter.addWidget(classes_panel)

        # Set Splitter Sizes (adjust ratios as needed)
        total_width = self.width()
        main_splitter.setSizes([int(total_width * 0.18), int(total_width * 0.60), int(total_width * 0.22)])
        # Set stretch factors for better resizing
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1) # Center panel gets extra space
        main_splitter.setStretchFactor(2, 0)

    def _create_connections(self):
        """Connect signals and slots."""
        # Top Controls
        self.add_images_btn.clicked.connect(self._on_add_images_clicked)
        self.select_model_btn.clicked.connect(self._on_select_model_clicked)
        self.process_btn.clicked.connect(self.processImageClicked) # Forward signal
        self.save_btn.clicked.connect(self._on_save_dataset_clicked)
        self.augment_btn.clicked.connect(self._on_augment_dataset_clicked)
        self.conf_spinbox.valueChanged.connect(self._on_config_changed)
        self.iou_spinbox.valueChanged.connect(self._on_config_changed)
        self.train_split_spinbox.valueChanged.connect(self._on_config_changed)

        # Image List
        self.images_list.currentItemChanged.connect(self._on_image_selected)

        # Canvas Controls
        self.zoom_in_btn.clicked.connect(self.image_canvas.zoom_in)
        self.zoom_out_btn.clicked.connect(self.image_canvas.zoom_out)
        self.fit_btn.clicked.connect(self.image_canvas.fit_to_view)
        self.draw_box_btn.toggled.connect(self._on_draw_box_toggled)
        self.delete_box_btn.clicked.connect(self.deleteSelectedBoxClicked) # Forward signal

        # Class List / Management
        self.classes_list.currentItemChanged.connect(self._on_class_selected)
        self.add_class_btn.clicked.connect(self._on_add_class_clicked)
        self.new_class_edit.returnPressed.connect(self._on_add_class_clicked)
        self.remove_class_btn.clicked.connect(self._on_remove_class_clicked)

        # Canvas -> Main Window (indirectly to AppLogic)
        self.image_canvas.boxSelected.connect(self._on_box_selected_in_canvas)

    # --- Event Handlers / Slots ---

    def _on_add_images_clicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if file_paths:
            self.addImagesClicked.emit(file_paths)

    def _on_select_model_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "YOLO Models (*.pt *.onnx)"
        )
        if file_path:
            self.selectModelClicked.emit(file_path)

    def _on_save_dataset_clicked(self):
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Dataset", ""
        )
        if output_dir:
            self.saveDatasetClicked.emit(output_dir)

    def _on_augment_dataset_clicked(self):
        num_augmentations = self.augmentations_spin.value()
        self.augmentDatasetClicked.emit(num_augmentations)

    def _on_add_class_clicked(self):
        class_name = self.new_class_edit.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Add Class", "Class name cannot be empty.")
            return
        # Check for duplicates
        items = self.classes_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if items:
            QMessageBox.warning(self, "Add Class", f"Class '{class_name}' already exists.")
            return

        self.addClassClicked.emit(class_name)
        self.new_class_edit.clear()

    def _on_remove_class_clicked(self):
        selected_item = self.classes_list.currentItem()
        if selected_item:
            class_name = selected_item.text()
            reply = QMessageBox.question(self, "Remove Class",
                                       f"Are you sure you want to remove class '{class_name}'?\n"
                                       "This will unassign it from all bounding boxes.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.removeClassClicked.emit(class_name)
        else:
             QMessageBox.warning(self, "Remove Class", "Please select a class to remove.")


    def _on_image_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        if current:
            image_path = current.data(Qt.ItemDataRole.UserRole)
            if image_path:
                 self.imageSelected.emit(image_path)
                 self.draw_box_btn.setChecked(False) # Disable drawing when switching images
        else:
             # Handle case where list becomes empty or selection cleared
             self.imageSelected.emit("") # Send empty path perhaps? Or handle in AppLogic
             self.draw_box_btn.setChecked(False)
        self._update_buttons_state()


    def _on_class_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        selected_class_name = current.text() if current else None
        self.classSelected.emit(selected_class_name or "") # Emit name or empty string

        # Highlight selected item visually (optional, but good UX)
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            is_selected = item is current
            font = item.font()
            font.setBold(is_selected)
            item.setFont(font)
            # Change background color slightly (requires careful handling with themes)
            # item.setBackground(QColor(200, 220, 255) if is_selected else self.classes_list.palette().base())

        self._update_buttons_state()


    def _on_box_selected_in_canvas(self, box_index: int):
        """Called when ImageCanvas signals a box selection change."""
        # No direct action needed here usually, AppLogic handles the assignment logic
        # based on this event AND the currently selected class.
        # We just need to update button states.
        self._update_buttons_state()


    def _on_config_changed(self):
        """Emit current configuration values."""
        config = self.get_config()
        self.configChanged.emit(config)

    def _on_draw_box_toggled(self, checked: bool):
        """Handle Draw Box button toggle."""
        self.image_canvas.set_drawing_enabled(checked)
        self.drawBoxClicked.emit(checked) # Inform AppLogic
        # Disable selection list interaction while drawing
        self.images_list.setEnabled(not checked)
        self.classes_list.setEnabled(not checked)
        if checked:
             self.image_canvas.select_box(-1) # Deselect any box
        self._update_buttons_state()


    # --- Public Methods for AppLogic to Call ---

    def add_images_to_list(self, image_paths: list[str]):
        current_paths = {self.images_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.images_list.count())}
        added_count = 0
        for path in image_paths:
            if path not in current_paths:
                item_text = os.path.basename(path)
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, path) # Store full path
                self.images_list.addItem(item)
                added_count += 1

        self.images_status_label.setText(f"Loaded: {self.images_list.count()} images")
        if added_count > 0 and self.images_list.currentRow() == -1:
             self.images_list.setCurrentRow(0) # Select first image if nothing was selected
        self._update_buttons_state()


    def set_model_path(self, model_path: str):
        model_name = os.path.basename(model_path) if model_path else "None"
        self.model_label.setText(f"Selected Model: {model_name}")
        self.model_label.setToolTip(model_path or "No model selected")
        self._update_buttons_state()

    def add_class_to_list(self, class_name: str):
        # AppLogic should ensure it's not a duplicate, but double-check here
        items = self.classes_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if not items:
            item = QListWidgetItem(class_name)
            self.classes_list.addItem(item)
            self.classes_list.setCurrentItem(item) # Select the newly added class
        self._update_buttons_state()


    def remove_class_from_list(self, class_name: str):
        items = self.classes_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        for item in items:
            self.classes_list.takeItem(self.classes_list.row(item))
        # Clear selection if the removed item was selected
        if not self.classes_list.currentItem():
             self._on_class_selected(None, None) # Trigger selection update
        self._update_buttons_state()


    def get_selected_image_path(self) -> str | None:
        selected_item = self.images_list.currentItem()
        return selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None

    def get_selected_class_name(self) -> str | None:
        selected_item = self.classes_list.currentItem()
        return selected_item.text() if selected_item else None

    def get_class_names(self) -> list[str]:
        return [self.classes_list.item(i).text() for i in range(self.classes_list.count())]

    def display_image(self, image_path: str | None):
        if image_path and os.path.exists(image_path):
            success = self.image_canvas.set_image(image_path=image_path)
            if not success:
                QMessageBox.warning(self, "Image Load Error", f"Failed to load image:\n{image_path}")
                # Optionally remove from list or mark as bad
        else:
            self.image_canvas.set_image() # Clear canvas if path is invalid/None
        self._update_buttons_state()


    def update_annotations(self, annotations: list[dict]):
        self.image_canvas.set_annotations(annotations)
        self._update_buttons_state()


    def update_class_names_display(self, class_names: list[str]):
        """Update both the class list widget and the canvas's knowledge."""
        self.image_canvas.set_class_names(class_names)

        # --- Sync QListWidget ---
        current_selection = self.get_selected_class_name()
        # Remember scroll position
        scroll_bar = self.classes_list.verticalScrollBar()
        scroll_pos = scroll_bar.value()

        self.classes_list.currentItemChanged.disconnect(self._on_class_selected) # Avoid signals during update
        self.classes_list.clear()
        self.classes_list.addItems(class_names)

        # Restore selection if possible
        new_selection_item = None
        if current_selection in class_names:
            items = self.classes_list.findItems(current_selection, Qt.MatchFlag.MatchExactly)
            if items:
                 new_selection_item = items[0]
                 self.classes_list.setCurrentItem(new_selection_item)

        # Restore scroll position
        scroll_bar.setValue(scroll_pos)
        self.classes_list.currentItemChanged.connect(self._on_class_selected)

        # Manually trigger selection update if selection changed (or cleared)
        self._on_class_selected(new_selection_item, None) # Call manually to update highlight/state

        self._update_buttons_state()


    def set_progress(self, value: int, max_value: int = 100):
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(value > 0 and value < max_value)

    def set_status(self, message: str, timeout: int = 0):
        self.status_bar.showMessage(message, timeout)

    def show_message(self, title: str, message: str, level: str = "info"):
        icon = QMessageBox.Icon.Information
        if level == "warning":
            icon = QMessageBox.Icon.Warning
        elif level == "error":
            icon = QMessageBox.Icon.Critical
        QMessageBox(icon, title, message, QMessageBox.StandardButton.Ok, self).exec()

    def get_config(self) -> dict:
        """Return current configuration settings."""
        return {
            'conf_threshold': self.conf_spinbox.value(),
            'iou_threshold': self.iou_spinbox.value(),
            'train_split': self.train_split_spinbox.value(),
        }

    def enable_processing_controls(self, enable: bool):
        """Enable/disable controls during long operations."""
        self.add_images_btn.setEnabled(enable)
        self.select_model_btn.setEnabled(enable)
        self.process_btn.setEnabled(enable)
        self.augment_btn.setEnabled(enable)
        self.save_btn.setEnabled(enable)
        self.images_list.setEnabled(enable)
        self.classes_list.setEnabled(enable)
        # Keep canvas controls always enabled? Or disable some?
        self.draw_box_btn.setEnabled(enable)
        self.delete_box_btn.setEnabled(enable and self.image_canvas.selected_box_idx != -1)

    # --- Internal State Update ---

    def _update_buttons_state(self):
        """Update the enabled state of buttons based on current application state."""
        has_images = self.images_list.count() > 0
        has_model = not self.model_label.text().endswith("None")
        has_classes = self.classes_list.count() > 0
        image_selected = self.get_selected_image_path() is not None
        class_selected = self.get_selected_class_name() is not None
        box_selected = self.image_canvas.selected_box_idx != -1
        is_drawing = self.draw_box_btn.isChecked()

        # Process button needs image and model
        self.process_btn.setEnabled(image_selected and has_model and not is_drawing)

        # Class removal needs a selected class
        self.remove_class_btn.setEnabled(class_selected and not is_drawing)

        # Augment/Save buttons require processed data (AppLogic controls these via explicit signals)
        # self.augment_btn.setEnabled(...) # Controlled by AppLogic
        # self.save_btn.setEnabled(...) # Controlled by AppLogic

        # Canvas controls
        canvas_exists = self.image_canvas.pixmap is not None
        self.zoom_in_btn.setEnabled(canvas_exists)
        self.zoom_out_btn.setEnabled(canvas_exists)
        self.fit_btn.setEnabled(canvas_exists)
        self.draw_box_btn.setEnabled(canvas_exists) # Enable draw if image loaded
        self.delete_box_btn.setEnabled(canvas_exists and box_selected and not is_drawing)

    # Style Sheet / Paint Event for minor tweaks if needed
    # def paintEvent(self, event):
    #     opt = QStyleOption()
    #     opt.initFrom(self)
    #     p = QPainter(self)
    #     self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, p, self)