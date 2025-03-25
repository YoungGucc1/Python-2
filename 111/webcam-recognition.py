import sys
import os
import sqlite3
import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QListWidget, QWidget, QStatusBar, 
                             QGroupBox, QTextEdit, QFileDialog, QInputDialog, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTime

class WebcamThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.paused = False
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while self.running:
            if not self.paused:
                ret, frame = cap.read()
                if ret:
                    self.update_frame.emit(frame)
            self.msleep(30)  # ~30 fps
        
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self):
        self.paused = not self.paused
        return self.paused

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setMinimumSize(1000, 700)
        
        # Database path
        self.db_path = None
        
        # Face recognition variables
        self.face_threshold = 0.7  # Similarity threshold
        self.known_faces = {}  # Will store name:embedding pairs
        self.current_face_embedding = None
        self.current_face_name = None
        
        # Setup UI first
        self.setup_ui()
        
        # Initialize models after UI is set up
        self.init_models()
        
        # Initialize webcam thread but don't start it automatically
        self.webcam_thread = WebcamThread()
        self.webcam_thread.update_frame.connect(self.update_frame)
        # The webcam will be started when the user clicks the toggle_camera button
        
        # Recognition timer
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.process_current_frame)
        self.recognition_timer.start(500)  # Process every 500ms to avoid overload
    
    def init_models(self):
        # Initialize face detection and recognition models
        self.log_message("Initializing face detection model...")
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.log_message("Initializing face recognition model...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.facenet = self.facenet.cuda()
            self.log_message("Using GPU acceleration")
        else:
            self.log_message("Using CPU for processing")
    
    def setup_ui(self):
        # Set the application style with a new color scheme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 14px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:pressed {
                background-color: #89dceb;
            }
            QPushButton:disabled {
                background-color: #6c7086;
                color: #313244;
            }
            QListWidget, QTextEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
            }
            QGroupBox {
                color: #cdd6f4;
                font-weight: bold;
                border: 1px solid #313244;
                border-radius: 4px;
                margin-top: 1em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Camera view and controls
        left_panel = QVBoxLayout()
        
        # Camera view
        camera_group = QGroupBox("Camera View")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel("Camera Off - Click 'Start Camera' to begin")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #181825; border: 2px solid #313244; border-radius: 4px; color: #cdd6f4; font-size: 16px;")
        self.camera_label.setMinimumSize(640, 480)
        
        camera_layout.addWidget(self.camera_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.toggle_camera_btn = QPushButton("Start Camera")
        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        
        self.capture_face_btn = QPushButton("Capture Face")
        self.capture_face_btn.clicked.connect(self.capture_face)
        self.capture_face_btn.setStyleSheet("background-color: #f5c2e7;")
        self.capture_face_btn.setEnabled(False)  # Disabled initially
        
        camera_controls.addWidget(self.toggle_camera_btn)
        camera_controls.addWidget(self.capture_face_btn)
        
        camera_layout.addLayout(camera_controls)
        camera_group.setLayout(camera_layout)
        
        # Face recognition status
        recognition_group = QGroupBox("Recognition Status")
        recognition_layout = QVBoxLayout()
        
        self.recognition_label = QLabel("No face detected")
        self.recognition_label.setAlignment(Qt.AlignCenter)
        self.recognition_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        
        recognition_layout.addWidget(self.recognition_label)
        recognition_group.setLayout(recognition_layout)
        
        # Add panels to left layout
        left_panel.addWidget(camera_group, 7)
        left_panel.addWidget(recognition_group, 1)
        
        # Right panel - Database and logs
        right_panel = QVBoxLayout()
        
        # Database controls
        db_group = QGroupBox("Database")
        db_layout = QVBoxLayout()
        
        self.db_path_label = QLabel("No database selected")
        self.db_path_label.setStyleSheet("font-style: italic;")
        
        db_buttons = QHBoxLayout()
        
        select_db_btn = QPushButton("Select Database")
        select_db_btn.clicked.connect(self.select_database)
        
        create_db_btn = QPushButton("Create Database")
        create_db_btn.clicked.connect(self.create_database)
        
        load_faces_btn = QPushButton("Load Known Faces")
        load_faces_btn.clicked.connect(self.load_known_faces)
        
        db_buttons.addWidget(select_db_btn)
        db_buttons.addWidget(create_db_btn)
        db_layout.addWidget(self.db_path_label)
        db_layout.addLayout(db_buttons)
        db_layout.addWidget(load_faces_btn)
        
        db_group.setLayout(db_layout)
        
        # Face management
        face_group = QGroupBox("Face Management")
        face_layout = QVBoxLayout()
        
        self.face_list = QListWidget()
        self.face_list.itemClicked.connect(self.face_selected)
        
        face_buttons = QHBoxLayout()
        
        add_face_btn = QPushButton("Add Current Face")
        add_face_btn.clicked.connect(self.add_face_to_db)
        add_face_btn.setStyleSheet("background-color: #a6e3a1;")
        
        delete_face_btn = QPushButton("Delete Selected")
        delete_face_btn.clicked.connect(self.delete_face)
        delete_face_btn.setStyleSheet("background-color: #f38ba8;")
        
        face_buttons.addWidget(add_face_btn)
        face_buttons.addWidget(delete_face_btn)
        
        face_layout.addWidget(QLabel("Known Faces:"))
        face_layout.addWidget(self.face_list)
        face_layout.addLayout(face_buttons)
        
        face_group.setLayout(face_layout)
        
        # Log section
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # Add panels to right layout
        right_panel.addWidget(db_group, 2)
        right_panel.addWidget(face_group, 3)
        right_panel.addWidget(log_group, 3)
        
        # Add both panels to main layout
        main_layout.addLayout(left_panel, 6)
        main_layout.addLayout(right_panel, 4)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initial log message
        self.log_message("Application started")
        self.log_message("Please select or create a database")
    
    def update_frame(self, frame):
        # Process the frame with MTCNN to get detected faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(Image.fromarray(frame_rgb))
        
        # If faces found, draw boxes
        if boxes is not None:
            for box in boxes:
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Convert to QImage and display
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
        self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.camera_label.size(), Qt.KeepAspectRatio))
    
    def process_current_frame(self):
        # Get the current frame
        pixmap = self.camera_label.pixmap()
        if pixmap is None:
            return
        
        qimg = pixmap.toImage()
        buffer = qimg.bits().asstring(qimg.byteCount())
        image = np.frombuffer(buffer, dtype=np.uint8).reshape((qimg.height(), qimg.width(), 4))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(image_rgb)
        
        # Detect faces
        try:
            face_tensor = self.mtcnn(pil_img)
            
            if face_tensor is None:
                self.recognition_label.setText("No face detected")
                self.recognition_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #f38ba8;")
                return
            
            # Get embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor.unsqueeze(0))
                self.current_face_embedding = embedding.cpu().numpy().flatten()
            
            # If we have known faces, compare and find the best match
            if self.known_faces and self.db_path:
                best_match = None
                best_score = 0
                
                for name, stored_embedding in self.known_faces.items():
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(self.current_face_embedding, stored_embedding)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = name
                
                if best_score > self.face_threshold:
                    self.current_face_name = best_match
                    self.recognition_label.setText(f"Recognized: {best_match}\nConfidence: {best_score:.2f}")
                    self.recognition_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #a6e3a1;")
                else:
                    self.current_face_name = None
                    self.recognition_label.setText(f"Unknown Face\nBest match: {best_match} ({best_score:.2f})")
                    self.recognition_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #f9e2af;")
            else:
                self.recognition_label.setText("Face detected\nNo database loaded")
                self.recognition_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #89b4fa;")
        
        except Exception as e:
            self.log_message(f"Error processing frame: {str(e)}")
    
    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def toggle_camera(self):
        # If the thread is not running yet, start it
        if not self.webcam_thread.isRunning():
            self.webcam_thread.start()
            self.toggle_camera_btn.setText("Pause Camera")
            self.status_bar.showMessage("Camera started")
            self.capture_face_btn.setEnabled(True)  # Enable capture button when camera starts
            return
            
        # If already running, toggle pause/resume
        paused = self.webcam_thread.pause()
        if paused:
            self.toggle_camera_btn.setText("Resume Camera")
            self.status_bar.showMessage("Camera paused")
            self.capture_face_btn.setEnabled(False)  # Disable capture button when camera is paused
        else:
            self.toggle_camera_btn.setText("Pause Camera")
            self.status_bar.showMessage("Camera resumed")
            self.capture_face_btn.setEnabled(True)  # Enable capture button when camera is resumed
    
    def capture_face(self):
        if self.current_face_embedding is not None:
            self.log_message("Face captured and ready to be added to database")
            self.status_bar.showMessage("Face captured")
        else:
            self.log_message("No face detected for capture")
    
    def select_database(self):
        db_path, _ = QFileDialog.getOpenFileName(
            self, "Select Database", "", "SQLite Database (*.db *.sqlite)"
        )
        if db_path:
            self.db_path = db_path
            self.db_path_label.setText(os.path.basename(db_path))
            self.status_bar.showMessage(f"Database: {db_path}")
            self.log_message(f"Selected database: {db_path}")
            
            self.ensure_database_structure()
            self.load_known_faces()
    
    def create_database(self):
        db_path, _ = QFileDialog.getSaveFileName(
            self, "Create Database", "", "SQLite Database (*.db *.sqlite)"
        )
        if db_path:
            self.db_path = db_path
            self.db_path_label.setText(os.path.basename(db_path))
            self.status_bar.showMessage(f"Database: {db_path}")
            self.log_message(f"Created database: {db_path}")
            
            self.create_database_structure()
    
    def ensure_database_structure(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                embedding BLOB
            )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.log_message(f"Database error: {str(e)}")
    
    def create_database_structure(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                embedding BLOB
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.log_message("Database structure created successfully")
        except Exception as e:
            self.log_message(f"Error creating database: {str(e)}")
    
    def load_known_faces(self):
        if not self.db_path:
            self.log_message("No database selected")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            
            self.known_faces = {}
            self.face_list.clear()
            
            for name, embedding_blob in rows:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                self.known_faces[name] = embedding
                self.face_list.addItem(name)
            
            conn.close()
            
            self.log_message(f"Loaded {len(self.known_faces)} faces from database")
            self.status_bar.showMessage(f"Loaded {len(self.known_faces)} faces")
        except Exception as e:
            self.log_message(f"Error loading faces: {str(e)}")
    
    def face_selected(self, item):
        self.status_bar.showMessage(f"Selected face: {item.text()}")
    
    def add_face_to_db(self):
        if self.current_face_embedding is None:
            self.log_message("No face detected to add")
            return
        
        if not self.db_path:
            self.log_message("No database selected")
            return
        
        name, ok = QInputDialog.getText(self, "Add Face", "Enter name for this face:", QLineEdit.Normal)
        
        if ok and name:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if the name already exists
                cursor.execute("SELECT id FROM face_embeddings WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing face
                    cursor.execute("UPDATE face_embeddings SET embedding = ? WHERE name = ?",
                                  (self.current_face_embedding.tobytes(), name))
                    self.log_message(f"Updated existing face: {name}")
                else:
                    # Insert new face
                    cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)",
                                  (name, self.current_face_embedding.tobytes()))
                    self.log_message(f"Added new face: {name}")
                
                conn.commit()
                conn.close()
                
                # Update known faces
                self.known_faces[name] = self.current_face_embedding
                
                # Update face list
                self.face_list.clear()
                for face_name in self.known_faces.keys():
                    self.face_list.addItem(face_name)
                
                self.status_bar.showMessage(f"Added face: {name}")
            except Exception as e:
                self.log_message(f"Error adding face: {str(e)}")
    
    def delete_face(self):
        if not self.face_list.currentItem():
            self.log_message("No face selected for deletion")
            return
        
        name = self.face_list.currentItem().text()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM face_embeddings WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            
            # Remove from known faces dictionary
            if name in self.known_faces:
                del self.known_faces[name]
            
            # Update face list
            self.face_list.takeItem(self.face_list.currentRow())
            
            self.log_message(f"Deleted face: {name}")
            self.status_bar.showMessage(f"Deleted face: {name}")
        except Exception as e:
            self.log_message(f"Error deleting face: {str(e)}")
    
    def log_message(self, message):
        self.log_text.append(f"[{QTime.currentTime().toString('hh:mm:ss')}] {message}")
    
    def closeEvent(self, event):
        # Stop the webcam thread before closing
        if hasattr(self, 'webcam_thread') and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 46))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(24, 24, 37))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(137, 180, 250))
    dark_palette.setColor(QPalette.Highlight, QColor(137, 180, 250))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())