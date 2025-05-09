import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QLineEdit, QStatusBar, QFileDialog, QMessageBox, QStyleOption, QStyle,
    QMenuBar, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QSettings, QCoreApplication, QDir, QPoint, QByteArray, pyqtSlot
from PyQt6.QtGui import QIcon, QPainter, QColor, QPalette, QAction, QCloseEvent

# Use local imports due to flat structure
from image_canvas import ImageCanvas
# from app_logic import AppLogic # Not needed for type hint here if only used in set_app_logic
from train_dialog import TrainDialog # Import the new dialog

class MainWindow(QMainWindow):
    """
    Main window for the YOLO Dataset Creator application.
    """
    # --- Signals for AppLogic ---
    addImagesClicked = pyqtSignal(list)         
    selectModelClicked = pyqtSignal(str)        
    processImageClicked = pyqtSignal()          
    saveDatasetClicked = pyqtSignal(str)        
    addClassClicked = pyqtSignal(str)           
    removeClassClicked = pyqtSignal(str)        
    imageSelected = pyqtSignal(str)             
    classSelected = pyqtSignal(str)             
    configChanged = pyqtSignal(dict)            
    drawBoxClicked = pyqtSignal(bool)           
    deleteSelectedBoxClicked = pyqtSignal()

    newProjectRequested = pyqtSignal()
    openProjectRequested = pyqtSignal(str) 
    saveProjectRequested = pyqtSignal()
    saveProjectAsRequested = pyqtSignal(str) 
    requestClose = pyqtSignal() 
    processAllImagesClicked = pyqtSignal() # New signal for processing all images
    trainModelParamsCollected = pyqtSignal(dict) # Signal for training parameters

    def __init__(self):
        super().__init__()
        self._app_logic = None 
        self.base_window_title = "YOLO Dataset Creator (PyQt6)"
        self.setWindowTitle(self.base_window_title)

        self._create_actions() 
        self._create_menu()
        self._create_widgets()
        self._create_layout()
        self._create_connections()
        self._update_buttons_state({}) 

        self.status_bar.showMessage("Ready")
        self._read_settings() 

    def set_app_logic(self, app_logic_instance): # Changed arg name for clarity
        """Set a reference to AppLogic, needed for closeEvent."""
        self._app_logic = app_logic_instance

    def _create_actions(self):
        """Create QAction objects for menus and potentially toolbars."""
        self.new_action = QAction(QIcon.fromTheme("document-new"), "&New Project", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setStatusTip("Create a new annotation project")
        self.new_action.triggered.connect(self._on_new_project)

        self.open_action = QAction(QIcon.fromTheme("document-open"), "&Open Project...", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setStatusTip("Open an existing annotation project (.ydc_proj file)")
        self.open_action.triggered.connect(self._prompt_open_project)

        self.save_action = QAction(QIcon.fromTheme("document-save"), "&Save Project", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setStatusTip("Save the current project")
        self.save_action.triggered.connect(self.saveProjectRequested)

        self.save_as_action = QAction(QIcon.fromTheme("document-save-as"), "Save Project &As...", self)
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.setStatusTip("Save the current project to a new file")
        self.save_as_action.triggered.connect(self._prompt_save_project_as)

        self.exit_action = QAction(QIcon.fromTheme("application-exit"), "E&xit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.triggered.connect(self.close) 

    def _create_menu(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

    def _create_widgets(self):
        """Create all the widgets for the main window."""
        self.add_images_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), " Add Images...")
        self.select_model_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), " Select Model...")
        self.model_label = QLabel("Selected Model: None")
        self.model_label.setWordWrap(True)

        self.process_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), " Process Image")
        self.process_all_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward), " Process All") # New Button
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0); self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.25); self.conf_spinbox.setPrefix("Conf: ")
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.0); self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45); self.iou_spinbox.setPrefix("IoU: ")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False) 

        self.augmentations_spin = QSpinBox()
        self.augmentations_spin.setRange(0, 50); self.augmentations_spin.setValue(0) 
        self.augmentations_spin.setPrefix("Aug Count: ") 
        self.augmentations_spin.setToolTip("Number of augmented variations per image to generate during Save Dataset")

        self.train_model_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), " Train Model...") # New button

        self.save_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), " Save Dataset...")
        self.train_split_spinbox = QDoubleSpinBox()
        self.train_split_spinbox.setRange(0.1, 0.9); self.train_split_spinbox.setSingleStep(0.05)
        self.train_split_spinbox.setValue(0.8); self.train_split_spinbox.setPrefix("Train Split: ")
        self.train_split_spinbox.setDecimals(2)

        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.images_status_label = QLabel("Loaded: 0 images")

        self.image_canvas = ImageCanvas()
        self.zoom_in_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), " Zoom +")
        self.zoom_out_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown), " Zoom -")
        self.fit_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), " Fit") 
        self.draw_box_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), " Draw Box") 
        self.draw_box_btn.setCheckable(True)
        self.delete_box_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), " Delete Box")

        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("Enter new class name...")
        self.add_class_btn = QPushButton("Add Class")
        self.remove_class_btn = QPushButton("Remove Selected")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _create_layout(self):
        """Create the main layout and add widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        controls_panel = QWidget()
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.addWidget(self.add_images_btn)
        controls_layout.addWidget(self.select_model_btn)
        controls_layout.addWidget(self.model_label, 1) 
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(self.process_all_btn)
        controls_layout.addWidget(self.conf_spinbox); controls_layout.addWidget(self.iou_spinbox)
        controls_layout.addWidget(self.progress_bar, 1) 
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.train_model_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.augmentations_spin); controls_layout.addWidget(self.train_split_spinbox)
        main_layout.addWidget(controls_panel)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter, 1) 

        images_panel = QWidget(); images_layout = QVBoxLayout(images_panel)
        images_layout.addWidget(QLabel("<b>Image Files</b>")); images_layout.addWidget(self.images_list, 1)
        images_layout.addWidget(self.images_status_label)
        main_splitter.addWidget(images_panel)

        annotation_panel = QWidget(); annotation_layout = QVBoxLayout(annotation_panel)
        annotation_layout.addWidget(QLabel("<b>Annotation Area</b>"))
        annotation_layout.addWidget(self.image_canvas, 1) 

        canvas_controls_layout = QHBoxLayout()
        canvas_controls_layout.addWidget(self.zoom_in_btn); canvas_controls_layout.addWidget(self.zoom_out_btn)
        canvas_controls_layout.addWidget(self.fit_btn); canvas_controls_layout.addStretch(1)
        canvas_controls_layout.addWidget(self.draw_box_btn); canvas_controls_layout.addWidget(self.delete_box_btn)
        annotation_layout.addLayout(canvas_controls_layout)

        instructions = QLabel("Ctrl+Scroll: Zoom | Middle Mouse/Ctrl+Drag: Pan | Click Box: Select | Click Class: Assign/Select")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter); instructions.setWordWrap(True)
        annotation_layout.addWidget(instructions)
        main_splitter.addWidget(annotation_panel)

        classes_panel = QWidget(); classes_layout = QVBoxLayout(classes_panel)
        classes_layout.addWidget(QLabel("<b>Classes</b>")); classes_layout.addWidget(self.classes_list, 1)
        classes_layout.addWidget(QLabel("Add New Class:")); classes_layout.addWidget(self.new_class_edit)
        add_remove_layout = QHBoxLayout()
        add_remove_layout.addWidget(self.add_class_btn); add_remove_layout.addWidget(self.remove_class_btn)
        classes_layout.addLayout(add_remove_layout)
        main_splitter.addWidget(classes_panel)

        total_width = self.width()
        main_splitter.setSizes([int(total_width * 0.18), int(total_width * 0.60), int(total_width * 0.22)])
        main_splitter.setStretchFactor(0, 0); main_splitter.setStretchFactor(1, 1); main_splitter.setStretchFactor(2, 0)

    def _create_connections(self):
        """Connect signals and slots."""
        self.add_images_btn.clicked.connect(self._on_add_images_clicked)
        self.select_model_btn.clicked.connect(self._on_select_model_clicked)
        self.process_btn.clicked.connect(self.processImageClicked)
        self.process_all_btn.clicked.connect(self.processAllImagesClicked)
        self.save_btn.clicked.connect(self._on_save_dataset_clicked)
        self.conf_spinbox.valueChanged.connect(self._on_config_changed)
        self.iou_spinbox.valueChanged.connect(self._on_config_changed)
        self.train_split_spinbox.valueChanged.connect(self._on_config_changed)
        self.augmentations_spin.valueChanged.connect(self._on_config_changed)

        self.images_list.currentItemChanged.connect(self._on_image_selected)

        self.zoom_in_btn.clicked.connect(self.image_canvas.zoom_in)
        self.zoom_out_btn.clicked.connect(self.image_canvas.zoom_out)
        self.fit_btn.clicked.connect(self.image_canvas.fit_to_view)
        self.draw_box_btn.toggled.connect(self._on_draw_box_toggled)
        self.delete_box_btn.clicked.connect(self.deleteSelectedBoxClicked) # Signal emitted by this button

        self.classes_list.currentItemChanged.connect(self._on_class_selected)
        self.add_class_btn.clicked.connect(self._on_add_class_clicked)
        self.new_class_edit.returnPressed.connect(self._on_add_class_clicked)
        self.remove_class_btn.clicked.connect(self._on_remove_class_clicked)

        self.image_canvas.boxSelected.connect(self._on_box_selected_in_canvas)

        self.train_model_btn.clicked.connect(self._on_train_model_button_clicked) # Connect train button

    # --- Event Handlers / Slots for UI Actions ---

    def _get_last_dir(self, key: str) -> str:
        settings = QSettings(); return settings.value(f"ui/last{key}Dir", QDir.homePath(), type=str)

    def _set_last_dir(self, key: str, dir_path: str):
        settings = QSettings(); settings.setValue(f"ui/last{key}Dir", dir_path)

    def _on_add_images_clicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", self._get_last_dir("Image"), "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if file_paths:
            self._set_last_dir("Image", os.path.dirname(file_paths[0]))
            self.addImagesClicked.emit(file_paths)

    def _on_select_model_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", self._get_last_dir("Model"), "YOLO Models (*.pt *.onnx)"
        )
        # Emit file_path even if it's empty (dialog cancelled), AppLogic will handle None/empty.
        self.selectModelClicked.emit(file_path if file_path else "")
        if file_path: # Only set last_dir if a file was actually selected
            self._set_last_dir("Model", os.path.dirname(file_path))


    def _on_save_dataset_clicked(self):
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Dataset", self._get_last_dir("Dataset")
        )
        if output_dir:
            self._set_last_dir("Dataset", output_dir)
            self.saveDatasetClicked.emit(output_dir)

    def _on_new_project(self):
        if self._app_logic:
            self._app_logic.new_project()

    def _prompt_open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", self._get_last_dir("Project"), "YOLO Dataset Creator Project (*.ydc_proj);;All Files (*)"
        )
        if file_path:
            self._set_last_dir("Project", os.path.dirname(file_path))
            self.openProjectRequested.emit(file_path)

    def _prompt_save_project_as(self) -> bool: # Return bool for AppLogic
        suggested_name = "untitled.ydc_proj"
        if self._app_logic and self._app_logic.current_project_path:
            suggested_name = os.path.basename(self._app_logic.current_project_path)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", os.path.join(self._get_last_dir("Project"), suggested_name),
            "YOLO Dataset Creator Project (*.ydc_proj);;All Files (*)"
        )
        if file_path:
            if not file_path.lower().endswith(".ydc_proj"):
                 file_path += ".ydc_proj"
            self._set_last_dir("Project", os.path.dirname(file_path))
            self.saveProjectAsRequested.emit(file_path)
            return True 
        return False 

    def _on_add_class_clicked(self):
        class_name = self.new_class_edit.text().strip()
        if class_name:
            self.addClassClicked.emit(class_name)
            self.new_class_edit.clear()
        else:
            self.status_bar.showMessage("Class name cannot be empty.", 3000)

    def _on_remove_class_clicked(self):
        selected_items = self.classes_list.selectedItems()
        if selected_items:
            class_name = selected_items[0].text()
            reply = QMessageBox.question(self, "Remove Class",
                                         f"Are you sure you want to remove class '{class_name}'?\nThis will unassign it from all annotations.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                 self.removeClassClicked.emit(class_name)
        else:
             self.status_bar.showMessage("Select a class to remove.", 3000)

    def _on_image_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        image_path = current.data(Qt.ItemDataRole.UserRole) if current else ""
        self.imageSelected.emit(image_path)

    def _on_class_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        class_name = current.text() if current else ""
        self.classSelected.emit(class_name)

    def _on_box_selected_in_canvas(self, box_index: int):
        pass 

    def _on_config_changed(self):
        self.configChanged.emit(self.get_config())

    def _on_draw_box_toggled(self, checked: bool):
        self.drawBoxClicked.emit(checked)

    def _on_train_model_button_clicked(self):
        current_yaml_path = None
        if self._app_logic and hasattr(self._app_logic, 'get_current_dataset_yaml_path'):
            current_yaml_path = self._app_logic.get_current_dataset_yaml_path()
            if not current_yaml_path:
                self.show_message("Dataset Not Ready", 
                                  "The dataset YAML file path is not available. Please ensure your dataset is saved (File > Save Dataset) first.", 
                                  "warning")
                # We could prevent the dialog from opening, or let the dialog show the error.
                # For now, let the dialog open and show the N/A message / handle validation.

        dialog = TrainDialog(self, current_dataset_yaml_path=current_yaml_path)
        if dialog.exec():
            params = dialog.get_parameters()
            if params:
                self.status_bar.showMessage("Training parameters collected. Requesting training start...", 3000)
                self.trainModelParamsCollected.emit(params)
            else:
                # This case should ideally be handled by dialog validation preventing OK
                self.status_bar.showMessage("Failed to get valid training parameters.", 3000)
        else:
            self.status_bar.showMessage("Training parameter setup cancelled.", 3000)

    # --- UI Update Methods ---

    def add_images_to_list(self, image_paths: list[str]):
        current_paths = {self.images_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.images_list.count())}
        for path in image_paths:
            if path not in current_paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path) 
                item.setToolTip(path) 
                self.images_list.addItem(item)
        self.images_status_label.setText(f"Loaded: {self.images_list.count()} images")

    def set_model_path(self, model_path: str | None):
        if model_path and os.path.isfile(model_path):
            self.model_label.setText(f"Selected Model: {os.path.basename(model_path)}")
            self.model_label.setToolTip(model_path)
        else:
            self.model_label.setText("Selected Model: None")
            self.model_label.setToolTip("")

    # add_class_to_list, remove_class_from_list are effectively handled by update_class_names_display

    def get_selected_image_path(self) -> str | None:
        selected_items = self.images_list.selectedItems()
        return selected_items[0].data(Qt.ItemDataRole.UserRole) if selected_items else None

    def get_selected_class_name(self) -> str | None:
        selected_items = self.classes_list.selectedItems()
        return selected_items[0].text() if selected_items else None

    def get_class_names(self) -> list[str]:
        return [self.classes_list.item(i).text() for i in range(self.classes_list.count())]
    
    def select_class_in_list(self, class_name_to_select: str):
        """Programmatically selects a class in the classes_list."""
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            if item.text() == class_name_to_select:
                self.classes_list.setCurrentItem(item)
                return
        self.classes_list.clearSelection() # Clear if not found


    def display_image(self, image_path: str | None):
        if image_path:
            if not self.image_canvas.set_image(image_path=image_path):
                 self.show_message("Image Load Error", f"Failed to load or display image:\n{image_path}", "error")
        else:
            self.image_canvas.set_image(None) 

    def update_annotations(self, annotations: list[dict]):
        self.image_canvas.set_annotations(annotations)

    def update_class_names_display(self, class_names: list[str]):
        selected_text = self.get_selected_class_name()
        self.classes_list.clear()
        new_selection_row = -1
        for i, name in enumerate(class_names):
            item = QListWidgetItem(name)
            self.classes_list.addItem(item)
            if name == selected_text: new_selection_row = i
        if new_selection_row != -1: self.classes_list.setCurrentRow(new_selection_row)
        else: self.classes_list.clearSelection()
        self.image_canvas.set_class_names(class_names)

    def set_progress(self, value: int, max_value: int = 100):
        self.progress_bar.setRange(0, max_value if max_value > 0 else 100) # Ensure max_value is positive
        self.progress_bar.setValue(value)
        self.progress_bar.setTextVisible(value > 0 and value < (max_value if max_value > 0 else 100))


    def set_status(self, message: str, timeout: int = 0):
        self.status_bar.showMessage(message, timeout)

    def show_message(self, title: str, message: str, level: str = "info"):
        msg_box = QMessageBox(self); msg_box.setWindowTitle(title); msg_box.setText(message)
        icon = QMessageBox.Icon.Information
        if level == "warning": icon = QMessageBox.Icon.Warning
        elif level == "error": icon = QMessageBox.Icon.Critical
        msg_box.setIcon(icon); msg_box.exec()

    def get_config(self) -> dict:
        return {
            'conf_threshold': self.conf_spinbox.value(), 'iou_threshold': self.iou_spinbox.value(),
            'train_split': self.train_split_spinbox.value(), 'augment_count': self.augmentations_spin.value(),
        }

    @pyqtSlot(dict)
    def _update_buttons_state(self, state: dict):
        """Update enabled state of buttons based on the state dict from AppLogic."""
        is_processing = state.get('is_processing', False) # Get explicitly
        self.add_images_btn.setEnabled(state.get('enable_add_images', True))
        self.select_model_btn.setEnabled(state.get('enable_select_model', True))
        self.process_btn.setEnabled(state.get('enable_process', False))
        self.process_all_btn.setEnabled(state.get('enable_process_all', False))
        self.save_btn.setEnabled(state.get('enable_save_dataset', False))
        self.add_class_btn.setEnabled(state.get('enable_add_class', True))
        self.remove_class_btn.setEnabled(state.get('enable_remove_class', False))
        self.delete_box_btn.setEnabled(state.get('enable_delete_box', False))
        self.draw_box_btn.setEnabled(state.get('enable_draw_box', False))

        self.train_model_btn.setEnabled(state.get('enable_train_model', False)) # Manage new button state

        self.save_action.setEnabled(state.get('enable_save_project', False))
        self.save_as_action.setEnabled(state.get('enable_save_project_as', False))
        self.new_action.setEnabled(not is_processing)
        self.open_action.setEnabled(not is_processing)

    @pyqtSlot(str, bool)
    def _update_window_title(self, project_path: str, is_dirty: bool):
        title = self.base_window_title
        if project_path != "Unsaved Project":
             title = f"{os.path.basename(project_path)} - {title}"
        if is_dirty: title += " *"
        self.setWindowTitle(title)

    def _write_settings(self):
        settings = QSettings()
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("state", self.saveState())
        if hasattr(self, 'main_splitter'): 
             settings.setValue("mainSplitter", self.main_splitter.saveState())
        settings.endGroup()
        settings.beginGroup("UI")
        settings.setValue("confThreshold", self.conf_spinbox.value())
        settings.setValue("iouThreshold", self.iou_spinbox.value())
        settings.setValue("trainSplit", self.train_split_spinbox.value())
        settings.setValue("augmentCount", self.augmentations_spin.value())
        settings.endGroup()

    def _read_settings(self):
        settings = QSettings()
        settings.beginGroup("MainWindow")
        geom = settings.value("geometry", QByteArray())
        if geom and isinstance(geom, QByteArray) and not geom.isEmpty(): self.restoreGeometry(geom)
        state = settings.value("state", QByteArray())
        if state and isinstance(state, QByteArray) and not state.isEmpty(): self.restoreState(state)
        if hasattr(self, 'main_splitter'):
            splitter_state = settings.value("mainSplitter", QByteArray())
            if splitter_state and isinstance(splitter_state, QByteArray) and not splitter_state.isEmpty():
                self.main_splitter.restoreState(splitter_state)
            else:
                 total_width = self.size().width() if self.size().width() > 200 else 1200
                 self.main_splitter.setSizes([int(total_width * 0.18), int(total_width * 0.60), int(total_width * 0.22)])
        settings.endGroup()
        settings.beginGroup("UI")
        self.conf_spinbox.setValue(settings.value("confThreshold", 0.25, type=float))
        self.iou_spinbox.setValue(settings.value("iouThreshold", 0.45, type=float))
        self.train_split_spinbox.setValue(settings.value("trainSplit", 0.8, type=float))
        self.augmentations_spin.setValue(settings.value("augmentCount", 0, type=int))
        settings.endGroup()
        self._on_config_changed()

    def closeEvent(self, event: QCloseEvent):
        if self._app_logic and self._app_logic._check_save_before_proceed():
            self._write_settings() 
            event.accept()
        else:
            event.ignore()