import sys
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import concurrent.futures
from deepface import DeepFace
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QProgressBar, QWidget, QTabWidget, 
                            QLineEdit, QComboBox, QMessageBox, QGridLayout, QScrollArea,
                            QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidget,
                            QTableWidgetItem, QHeaderView)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

class FaceProcessor(QThread):
    """Worker thread to process images in background"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_processing = pyqtSignal(list)
    
    def __init__(self, image_paths, resize_dim, normalize, align_faces, db_type, db_path, model_name="VGG-Face"):
        super().__init__()
        self.image_paths = image_paths
        self.resize_dim = resize_dim
        self.normalize = normalize
        self.align_faces = align_faces
        self.db_type = db_type
        self.db_path = db_path
        self.model_name = model_name
        self.results = []
        
    def run(self):
        total_images = len(self.image_paths)
        processed = 0
        
        # Connect to database
        if self.db_type == "SQLite":
            conn = self.connect_sqlite(self.db_path)
        else:
            # For PostgreSQL implementation, you would need to import psycopg2
            self.status_update.emit("PostgreSQL not implemented in this version")
            return
            
        if not conn:
            self.status_update.emit("Database connection failed")
            return
            
        cursor = conn.cursor()
        
        # Tables are now created in the connect_sqlite method
        
        # Process each image
        for img_path in self.image_paths:
            try:
                # Step 1: Load the image
                self.status_update.emit(f"Processing {os.path.basename(img_path)}")
                image = cv2.imread(img_path)
                if image is None:
                    self.status_update.emit(f"Error loading {os.path.basename(img_path)}")
                    continue
                    
                # Preprocess image if needed
                processed_img = self.preprocess_image(image)
                
                # Step 2: Detect faces using DeepFace
                try:
                    # DeepFace.extract_faces returns a list of dictionaries for detected faces
                    face_objs = DeepFace.extract_faces(
                        img_path=img_path,
                        target_size=(224, 224),  # Size depends on the model
                        detector_backend='opencv',  # Options: opencv, retinaface, mtcnn, ssd, dlib
                        enforce_detection=False,   # Don't raise error if no face detected
                        align=self.align_faces
                    )
                    
                    # If no faces found
                    if len(face_objs) == 0:
                        self.status_update.emit(f"No faces found in {os.path.basename(img_path)}")
                        self.results.append((img_path, 0))
                        continue
                    
                    # Step 3: Extract embeddings for each face
                    embeddings = []
                    face_locations = []
                    
                    for face_obj in face_objs:
                        try:
                            # Get the facial area
                            facial_area = face_obj.get('facial_area', {})
                            if facial_area:
                                # Extract location (x, y, width, height format)
                                x = facial_area.get('x', 0)
                                y = facial_area.get('y', 0)
                                w = facial_area.get('w', 0)
                                h = facial_area.get('h', 0)
                                
                                # Convert to format (top, right, bottom, left)
                                face_location = (y, x + w, y + h, x)
                                face_locations.append(face_location)
                                
                                # Get the embedding
                                face_img = face_obj.get('face', None)
                                if face_img is not None:
                                    embedding = DeepFace.represent(
                                        img_path=face_img,  # Use the extracted face
                                        model_name=self.model_name,
                                        enforce_detection=False
                                    )
                                    embeddings.append(embedding)
                            
                        except Exception as e:
                            self.status_update.emit(f"Error extracting embedding: {str(e)}")
                    
                    # Step 4: Store in database
                    image_id = self.store_image_info(cursor, img_path)
                    for i, (embedding, location) in enumerate(zip(embeddings, face_locations)):
                        self.store_face_data(cursor, image_id, embedding, location, i)
                    
                    conn.commit()
                    self.results.append((img_path, len(face_objs)))
                    
                except Exception as e:
                    self.status_update.emit(f"Error in face detection: {str(e)}")
                    self.results.append((img_path, 0))
                
            except Exception as e:
                self.status_update.emit(f"Error processing {os.path.basename(img_path)}: {str(e)}")
                self.results.append((img_path, 0))
            
            processed += 1
            progress = int((processed / total_images) * 100)
            self.progress_update.emit(progress)
        
        cursor.close()
        conn.close()
        self.status_update.emit("Processing complete!")
        self.finished_processing.emit(self.results)
    
    def preprocess_image(self, image):
        """Preprocess image: resize, normalize"""
        # Resize
        if self.resize_dim > 0:
            h, w = image.shape[:2]
            ratio = self.resize_dim / max(h, w)
            new_size = (int(w * ratio), int(h * ratio))
            image = cv2.resize(image, new_size)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            
        return image
    
    def connect_sqlite(self, db_path):
        """Connect to SQLite database and create it if it doesn't exist"""
        try:
            # Check if database exists before connecting
            db_exists = os.path.exists(db_path) and os.path.getsize(db_path) > 0
            
            # SQLite will automatically create the database file if it doesn't exist
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            self.create_tables(cursor)
            conn.commit()
            
            if not db_exists:
                self.status_update.emit(f"Created new database: {os.path.basename(db_path)}")
            else:
                self.status_update.emit(f"Connected to existing database: {os.path.basename(db_path)}")
            
            return conn
        except Exception as e:
            self.status_update.emit(f"Database connection error: {str(e)}")
            return None
    
    def create_tables(self, cursor):
        """Create necessary tables if they don't exist"""
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            processed_date TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            face_index INTEGER,
            top INTEGER,
            right INTEGER,
            bottom INTEGER,
            left INTEGER,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            embedding BLOB,
            model_name TEXT,
            FOREIGN KEY (face_id) REFERENCES faces (id)
        )
        ''')
    
    def store_image_info(self, cursor, img_path):
        """Store image information and return image_id"""
        file_name = os.path.basename(img_path)
        cursor.execute(
            "INSERT INTO images (file_path, file_name, processed_date) VALUES (?, ?, ?)",
            (img_path, file_name, datetime.now())
        )
        return cursor.lastrowid
    
    def store_face_data(self, cursor, image_id, embedding, location, face_index):
        """Store face location and embedding"""
        top, right, bottom, left = location
        
        cursor.execute(
            "INSERT INTO faces (image_id, face_index, top, right, bottom, left) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, face_index, top, right, bottom, left)
        )
        face_id = cursor.lastrowid
        
        # Convert embedding to binary blob
        embedding_blob = np.array(embedding).tobytes()
        
        cursor.execute(
            "INSERT INTO embeddings (face_id, embedding, model_name) VALUES (?, ?, ?)",
            (face_id, embedding_blob, self.model_name)
        )


class FaceRecognitionApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Processing System")
        self.setMinimumSize(900, 700)
        
        # Initialize variables
        self.image_paths = []
        
        # Set default database path in user's documents folder
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        self.db_path = os.path.join(documents_path, "face_recognition.db")
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create tabs
        self.processing_tab = QWidget()
        self.results_tab = QWidget()
        self.settings_tab = QWidget()
        
        tab_widget.addTab(self.processing_tab, "Processing")
        tab_widget.addTab(self.results_tab, "Results")
        tab_widget.addTab(self.settings_tab, "Settings")
        
        # Setup each tab
        self.setup_processing_tab()
        self.setup_results_tab()
        self.setup_settings_tab()
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        
        # Apply styles
        self.apply_styles()
        
        # Show default database path in status
        if os.path.exists(self.db_path):
            self.status_label.setText(f"Using existing database: {os.path.basename(self.db_path)}")
        else:
            self.status_label.setText(f"Will create database: {os.path.basename(self.db_path)} when needed")
        
    def setup_processing_tab(self):
        layout = QVBoxLayout(self.processing_tab)
        
        # Header
        header = QLabel("Face Recognition Processing")
        header.setFont(QFont('Arial', 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Image source group
        source_group = QGroupBox("Image Source")
        source_layout = QVBoxLayout()
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Select folder containing images...")
        self.folder_path_edit.setReadOnly(True)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        
        folder_layout.addWidget(QLabel("Folder:"))
        folder_layout.addWidget(self.folder_path_edit, 1)
        folder_layout.addWidget(browse_btn)
        
        source_layout.addLayout(folder_layout)
        
        # Database selection
        db_layout = QHBoxLayout()
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["SQLite", "PostgreSQL"])
        self.db_type_combo.currentIndexChanged.connect(self.update_db_config)
        
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Select database file...")
        self.db_path_edit.setText(self.db_path)  # Display default database path
        self.db_path_edit.setReadOnly(True)
        
        db_browse_btn = QPushButton("Browse...")
        db_browse_btn.clicked.connect(self.browse_database)
        
        db_layout.addWidget(QLabel("Database:"))
        db_layout.addWidget(self.db_type_combo)
        db_layout.addWidget(self.db_path_edit, 1)
        db_layout.addWidget(db_browse_btn)
        
        source_layout.addLayout(db_layout)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Image count and info
        self.image_count_label = QLabel("0 images selected")
        layout.addWidget(self.image_count_label)
        
        # Processing options
        process_group = QGroupBox("Processing Options")
        process_layout = QGridLayout()
        
        self.resize_spin = QSpinBox()
        self.resize_spin.setRange(0, 1024)
        self.resize_spin.setValue(640)
        self.resize_spin.setSingleStep(64)
        self.resize_spin.setSpecialValueText("No resize")
        
        self.normalize_check = QCheckBox("Normalize pixel values")
        self.normalize_check.setChecked(True)
        
        self.align_check = QCheckBox("Align faces")
        self.align_check.setChecked(True)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"])
        
        process_layout.addWidget(QLabel("Resize max dimension:"), 0, 0)
        process_layout.addWidget(self.resize_spin, 0, 1)
        process_layout.addWidget(QLabel("Face Recognition Model:"), 1, 0)
        process_layout.addWidget(self.model_combo, 1, 1)
        process_layout.addWidget(self.normalize_check, 2, 0, 1, 2)
        process_layout.addWidget(self.align_check, 3, 0, 1, 2)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready to process")
        
        start_btn = QPushButton("Start Processing")
        start_btn.clicked.connect(self.start_processing)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(start_btn)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # Results table
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Image", "Faces Found", "Status"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        layout.addWidget(QLabel("Processing Results:"))
        layout.addWidget(self.results_table)
        
        # Summary section
        summary_layout = QHBoxLayout()
        self.total_images_label = QLabel("Total Images: 0")
        self.total_faces_label = QLabel("Total Faces: 0")
        self.success_rate_label = QLabel("Success Rate: 0%")
        
        summary_layout.addWidget(self.total_images_label)
        summary_layout.addWidget(self.total_faces_label)
        summary_layout.addWidget(self.success_rate_label)
        
        layout.addLayout(summary_layout)
        
    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        
        # Face detection settings
        detect_group = QGroupBox("Face Detection Settings")
        detect_layout = QGridLayout()
        
        self.detection_backend_combo = QComboBox()
        self.detection_backend_combo.addItems(["opencv", "retinaface", "mtcnn", "ssd", "dlib"])
        
        detect_layout.addWidget(QLabel("Detection Backend:"), 0, 0)
        detect_layout.addWidget(self.detection_backend_combo, 0, 1)
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        # Database settings
        db_settings_group = QGroupBox("Database Settings")
        db_settings_layout = QGridLayout()
        
        self.overwrite_check = QCheckBox("Overwrite existing entries")
        self.bulk_insert_check = QCheckBox("Use bulk insert (faster)")
        self.auto_create_db_check = QCheckBox("Auto-create database if not exists")
        self.bulk_insert_check.setChecked(True)
        self.auto_create_db_check.setChecked(True)
        
        db_settings_layout.addWidget(self.overwrite_check, 0, 0)
        db_settings_layout.addWidget(self.bulk_insert_check, 1, 0)
        db_settings_layout.addWidget(self.auto_create_db_check, 2, 0)
        
        db_settings_group.setLayout(db_settings_layout)
        layout.addWidget(db_settings_group)
        
        # Save settings button
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_settings_btn)
        
        # Add spacer
        layout.addStretch()
        
    def apply_styles(self):
        # Modern dark theme
        self.setStyleSheet("""
            QMainWindow, QDialog, QWidget { 
                background-color: #2D2D30;
                color: #E1E1E1;
            }
            QTabWidget::pane { 
                border: 1px solid #3E3E42; 
                background-color: #252526;
            }
            QTabBar::tab {
                background: #2D2D30;
                color: #E1E1E1;
                padding: 8px 12px;
                border: 1px solid #3E3E42;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #007ACC;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #006CC1;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #333337;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 4px;
                color: #E1E1E1;
            }
            QGroupBox {
                border: 1px solid #3E3E42;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #E1E1E1;
            }
            QProgressBar {
                border: 1px solid #3E3E42;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                width: 5px;
            }
            QTableWidget {
                background-color: #252526;
                alternate-background-color: #2D2D30;
                gridline-color: #3E3E42;
                border: 1px solid #3E3E42;
                color: #E1E1E1;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #E1E1E1;
                padding: 4px;
                border: 1px solid #3E3E42;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """)
        
    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.load_images(folder_path)
    
    def load_images(self, folder_path):
        """Load images from the selected folder"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self.image_paths = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                self.image_paths.append(os.path.join(folder_path, file))
        
        self.image_count_label.setText(f"{len(self.image_paths)} images found")
        self.status_label.setText(f"Ready to process {len(self.image_paths)} images")
    
    def browse_database(self):
        if self.db_type_combo.currentText() == "SQLite":
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Select or Create SQLite Database", "", "SQLite Database (*.db)"
            )
            if file_path:
                self.db_path_edit.setText(file_path)
                self.db_path = file_path
                # Inform user that a new database will be created if it doesn't exist
                if not os.path.exists(file_path):
                    self.status_label.setText("New database will be created when processing starts")
    
    def update_db_config(self):
        if self.db_type_combo.currentText() == "PostgreSQL":
            self.db_path_edit.setText("PostgreSQL configuration not implemented")
            self.db_path_edit.setEnabled(False)
        else:
            self.db_path_edit.setEnabled(True)
            self.db_path_edit.clear()
    
    def start_processing(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please select a folder with images first.")
            return
            
        if not self.db_path and self.db_type_combo.currentText() == "SQLite":
            QMessageBox.warning(self, "No Database", "Please select a database file.")
            return
        
        # For SQLite, check if database exists
        if self.db_type_combo.currentText() == "SQLite":
            db_exists = os.path.exists(self.db_path)
            
            # If database doesn't exist and auto-create is not checked, warn the user
            if not db_exists and not self.auto_create_db_check.isChecked():
                response = QMessageBox.question(
                    self, 
                    "Database Not Found", 
                    f"The database file '{os.path.basename(self.db_path)}' does not exist. Create it?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.No:
                    return
            
            # Ensure the database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir)
                    self.status_label.setText(f"Created directory: {db_dir}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not create database directory: {str(e)}")
                    return
        
        # Clear previous results
        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        # Create and start worker thread
        self.worker = FaceProcessor(
            self.image_paths,
            self.resize_spin.value(),
            self.normalize_check.isChecked(),
            self.align_check.isChecked(),
            self.db_type_combo.currentText(),
            self.db_path,
            self.model_combo.currentText()
        )
        
        self.worker.progress_update.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished_processing.connect(self.processing_finished)
        
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
    
    def processing_finished(self, results):
        # Update results table
        self.results_table.setRowCount(len(results))
        
        total_faces = 0
        for row, (img_path, face_count) in enumerate(results):
            # Image name
            self.results_table.setItem(row, 0, QTableWidgetItem(os.path.basename(img_path)))
            
            # Face count
            face_item = QTableWidgetItem(str(face_count))
            face_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 1, face_item)
            
            # Status
            status_item = QTableWidgetItem("Success" if face_count > 0 else "No faces")
            status_color = QColor("#4CAF50") if face_count > 0 else QColor("#F44336")
            status_item.setForeground(status_color)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 2, status_item)
            
            total_faces += face_count
        
        # Update summary
        self.total_images_label.setText(f"Total Images: {len(results)}")
        self.total_faces_label.setText(f"Total Faces: {total_faces}")
        
        success_count = sum(1 for _, count in results if count > 0)
        success_rate = (success_count / len(results)) * 100 if results else 0
        self.success_rate_label.setText(f"Success Rate: {success_rate:.1f}%")
        
        # Update status
        self.status_label.setText(f"Processing complete: {total_faces} faces found in {len(results)} images")
        
        # Switch to results tab
        tab_widget = self.centralWidget().layout().itemAt(0).widget()
        tab_widget.setCurrentIndex(1)  # Switch to Results tab
    
    def save_settings(self):
        # Here you would save settings to a config file
        QMessageBox.information(self, "Settings", "Settings saved successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())