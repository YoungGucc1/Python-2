"""
Scanner View
Handles the user interface elements of the image scanner application.
"""
import os
from typing import List, Dict, Optional, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QSlider, QStatusBar, QScrollArea, QFrame, QGroupBox,
    QMenu, QMessageBox, QTabWidget, QSizePolicy, QPlainTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QAction, QIcon, QColor, QFont


class SimilarImagesWidget(QWidget):
    """Widget to display similar images"""
    delete_image_requested = pyqtSignal(str)
    open_image_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.similar_image_widgets = {}
    
    def init_ui(self):
        """Initialize the UI"""
        self.layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel("Similar Images")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.title_label)
        
        # Scroll area for similar images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container widget for similar images
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)
    
    def add_similar_group(self, original_path: str, similar_images: List[Dict]):
        """Add a group of similar images"""
        # Create group box
        group_box = QGroupBox(os.path.basename(original_path))
        group_layout = QVBoxLayout()
        
        # Add original image
        original_widget = ImageWidget(original_path, is_original=True)
        original_widget.delete_image_requested.connect(self.delete_image_requested)
        original_widget.open_image_requested.connect(self.open_image_requested)
        group_layout.addWidget(original_widget)
        
        # Add similar images
        for img in similar_images:
            similar_widget = ImageWidget(img["path"], similarity=img["score"])
            similar_widget.delete_image_requested.connect(self.delete_image_requested)
            similar_widget.open_image_requested.connect(self.open_image_requested)
            group_layout.addWidget(similar_widget)
            
            # Store reference
            self.similar_image_widgets[img["path"]] = similar_widget
        
        group_box.setLayout(group_layout)
        self.scroll_layout.addWidget(group_box)
    
    def clear_similar_images(self):
        """Clear all similar images"""
        # Remove all widgets
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Clear references
        self.similar_image_widgets = {}


class ImageWidget(QFrame):
    """Widget to display an image with actions"""
    delete_image_requested = pyqtSignal(str)
    open_image_requested = pyqtSignal(str)
    
    def __init__(self, image_path: str, is_original: bool = False, similarity: float = None):
        super().__init__()
        self.image_path = image_path
        self.is_original = is_original
        self.similarity = similarity
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        
        # Main layout
        layout = QHBoxLayout(self)
        
        # Image thumbnail
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(100, 100)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_thumbnail()
        layout.addWidget(self.thumbnail)
        
        # Image info
        info_layout = QVBoxLayout()
        
        # Image name
        self.name_label = QLabel(os.path.basename(self.image_path))
        self.name_label.setWordWrap(True)
        info_layout.addWidget(self.name_label)
        
        # Image path
        path_label = QLabel(self.image_path)
        path_label.setWordWrap(True)
        path_label.setStyleSheet("color: gray; font-size: 8pt;")
        info_layout.addWidget(path_label)
        
        # Similarity score
        if self.similarity is not None:
            similarity_label = QLabel(f"Similarity: {self.similarity:.2f}")
            similarity_label.setStyleSheet(f"color: {'green' if self.similarity > 0.9 else 'orange'};")
            info_layout.addWidget(similarity_label)
        
        # Original image indicator
        if self.is_original:
            original_label = QLabel("Original")
            original_label.setStyleSheet("color: blue; font-weight: bold;")
            info_layout.addWidget(original_label)
        
        # Add spacer
        info_layout.addStretch()
        
        # Actions
        action_layout = QHBoxLayout()
        
        # Open button
        open_button = QPushButton("Open")
        open_button.clicked.connect(self.open_image)
        action_layout.addWidget(open_button)
        
        # Delete button
        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("background-color: #ff6b6b;")
        delete_button.clicked.connect(self.delete_image)
        action_layout.addWidget(delete_button)
        
        info_layout.addLayout(action_layout)
        layout.addLayout(info_layout, 1)
    
    def load_thumbnail(self):
        """Load and display thumbnail"""
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.thumbnail.setPixmap(pixmap)
            else:
                self.thumbnail.setText("No Preview")
        except Exception:
            self.thumbnail.setText("Error")
    
    def open_image(self):
        """Open the image"""
        self.open_image_requested.emit(self.image_path)
    
    def delete_image(self):
        """Delete the image"""
        reply = QMessageBox.question(
            self, 'Delete Image',
            f"Are you sure you want to delete this image?\n{self.image_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_image_requested.emit(self.image_path)
            self.setVisible(False)


class LogWidget(QPlainTextEdit):
    """Widget to display log messages"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setMaximumBlockCount(int(500))  # Limit number of lines
    
    def log(self, message: str, level: str = "info"):
        """Add a log message with formatting"""
        color_map = {
            "info": "black",
            "warning": "orange",
            "error": "red",
            "success": "green"
        }
        color = color_map.get(level, "black")
        
        # Format timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Add formatted message
        self.appendHtml(f"<span style='color:gray;'>[{timestamp}]</span> <span style='color:{color};'>{message}</span>")
        
        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MainWindow(QMainWindow):
    """Main window for the Image Scanner application"""
    
    # Signals for communication with controller
    start_scan_requested = pyqtSignal(str)
    stop_scan_requested = pyqtSignal()
    delete_image_requested = pyqtSignal(str)
    open_image_requested = pyqtSignal(str)
    similarity_threshold_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.database_path = ""
        self.folder_path = ""
    
    def init_ui(self):
        """Initialize the UI"""
        # Set window properties
        self.setWindowTitle("Image Scanner")
        self.setMinimumSize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top panel for controls
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        
        # Folder selection
        folder_layout = QVBoxLayout()
        folder_label = QLabel("Scan Folder:")
        folder_layout.addWidget(folder_label)
        
        folder_input_layout = QHBoxLayout()
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("font-style: italic;")
        folder_input_layout.addWidget(self.folder_path_label, 1)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_folder)
        folder_input_layout.addWidget(browse_button)
        
        folder_layout.addLayout(folder_input_layout)
        top_layout.addLayout(folder_layout)
        
        # Scan controls
        scan_layout = QVBoxLayout()
        scan_label = QLabel("Scan Controls:")
        scan_layout.addWidget(scan_label)
        
        scan_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Scan")
        self.start_button.clicked.connect(self.start_scan)
        scan_buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Scan")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        scan_buttons_layout.addWidget(self.stop_button)
        
        scan_layout.addLayout(scan_buttons_layout)
        top_layout.addLayout(scan_layout)
        
        # Similarity threshold
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel("Similarity Threshold:")
        threshold_layout.addWidget(threshold_label)
        
        threshold_slider_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(85)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_slider_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("0.85")
        threshold_slider_layout.addWidget(self.threshold_value_label)
        
        threshold_layout.addLayout(threshold_slider_layout)
        top_layout.addLayout(threshold_layout)
        
        main_layout.addWidget(top_panel)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        main_layout.addLayout(progress_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Similar images
        self.similar_images_widget = SimilarImagesWidget()
        self.similar_images_widget.delete_image_requested.connect(self.delete_image_requested)
        self.similar_images_widget.open_image_requested.connect(self.open_image_requested)
        splitter.addWidget(self.similar_images_widget)
        
        # Right side - Log panel
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        
        log_label = QLabel("Log:")
        log_layout.addWidget(log_label)
        
        self.log_widget = LogWidget()
        log_layout.addWidget(self.log_widget)
        
        splitter.addWidget(log_panel)
        
        # Set default sizes
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status bar label for database path
        self.db_path_label = QLabel()
        self.status_bar.addPermanentWidget(self.db_path_label)
        
        # Set initial status
        self.update_status("Ready")
        
        # Add menu bar
        self.create_menu_bar()
        
        # Initial log message
        self.log_message("Image Scanner started", "info")
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # Browse action
        browse_action = QAction("Browse Folder", self)
        browse_action.triggered.connect(self.browse_folder)
        file_menu.addAction(browse_action)
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Scan menu
        scan_menu = menu_bar.addMenu("Scan")
        
        # Start scan action
        start_action = QAction("Start Scan", self)
        start_action.triggered.connect(self.start_scan)
        scan_menu.addAction(start_action)
        
        # Stop scan action
        stop_action = QAction("Stop Scan", self)
        stop_action.triggered.connect(self.stop_scan)
        scan_menu.addAction(stop_action)
    
    def browse_folder(self):
        """Browse for a folder to scan"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if folder_path:
            self.folder_path = folder_path
            self.folder_path_label.setText(folder_path)
            self.log_message(f"Selected folder: {folder_path}", "info")
    
    def start_scan(self):
        """Start the scan process"""
        if not self.folder_path:
            self.log_message("No folder selected!", "error")
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_status("Scanning...")
        
        # Emit signal to start scan
        self.start_scan_requested.emit(self.folder_path)
    
    def stop_scan(self):
        """Stop the scan process"""
        self.stop_scan_requested.emit()
        self.update_status("Stopping...")
    
    def update_threshold_label(self, value):
        """Update the threshold label"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self.similarity_threshold_changed.emit(threshold)
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.showMessage(message)
    
    def set_database_path(self, path):
        """Set the database path"""
        self.database_path = path
        self.db_path_label.setText(f"Database: {os.path.basename(path)}")
    
    def set_similarity_threshold(self, value):
        """Set the similarity threshold"""
        self.threshold_slider.setValue(int(value * 100))
    
    def add_similar_images(self, original_path, similar_images):
        """Add similar images to the view"""
        self.similar_images_widget.add_similar_group(original_path, similar_images)
    
    def clear_similar_images(self):
        """Clear all similar images"""
        self.similar_images_widget.clear_similar_images()
    
    def update_progress(self, processed, total, speed):
        """Update the progress bar"""
        if total > 0:
            percent = (processed / total) * 100
            self.progress_bar.setValue(int(percent))
            self.progress_label.setText(f"{processed}/{total} ({percent:.1f}%) - {speed:.1f} images/sec")
    
    def scan_completed(self):
        """Handle scan completion"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Scan completed")
    
    def log_message(self, message, level="info"):
        """Add a log message"""
        self.log_widget.log(message, level)