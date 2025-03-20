import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QFileDialog, QProgressBar, 
                            QScrollArea, QGridLayout, QMessageBox, QTabWidget, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
from transformers import pipeline
import torch

# List of supported models - just the model IDs from the original
MODELS = [
    "google/vit-base-patch16-224",
    "nateraw/vit-age-classifier",
    "microsoft/resnet-50",
    "Falconsai/nsfw_image_detection",
    "cafeai/cafe_aesthetic",
    "microsoft/resnet-18",
    "microsoft/resnet-34",
    "microsoft/resnet-101",
    "microsoft/resnet-152",
    "microsoft/swin-tiny-patch4-window7-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "facebook/convnext-large-224",
    "timm/resnet50.a1_in1k",
    "timm/mobilenetv3_large_100.ra_in1k",
    "trpakov/vit-face-expression",
    "rizvandwiki/gender-classification",
    "LukeJacob2023/nsfw-image-detector",
    "vit-base-patch16-224-in21k",
    "not-lain/deepfake",
    "carbon225/vit-base-patch16-224-hentai",
    "facebook/convnext-base-224-22k-1k",
    "facebook/convnext-tiny-224",
    "nvidia/mit-b0",
    "microsoft/swinv2-base-patch4-window16-256",
    "andupets/real-estate-image-classification",
    "timm/tf_efficientnetv2_s.in21k",
    "timm/convnext_tiny.fb_in22k",
    "DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition",
    "FatihC/swin-tiny-patch4-window7-224-finetuned-eurosat-watermark",
    "aalonso-developer/vit-base-patch16-224-in21k-clothing-classifier",
    "RickyIG/emotion_face_image_classification",
    "shadowlilac/aesthetic-shadow"
]

MIN_ACCEPTABLE_SCORE = 0.1
MAX_N_LABELS = 5

class ClassificationWorker(QThread):
    """Worker thread to handle image classification to keep UI responsive"""
    progress_update = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(list)
    finished_all = pyqtSignal()
    
    def __init__(self, model_path, image_paths, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.image_paths = image_paths
        self.results = []
        
    def run(self):
        try:
            # Initialize the classifier
            classifier = pipeline("image-classification", model=self.model_path)
            
            for i, img_path in enumerate(self.image_paths):
                try:
                    image = Image.open(img_path)
                    result = classifier(image)
                    
                    # Add filename to results
                    result_with_filename = {
                        'filename': os.path.basename(img_path),
                        'path': img_path,
                        'classifications': result
                    }
                    self.results.append(result_with_filename)
                    
                    # Emit progress
                    self.progress_update.emit(i + 1, len(self.image_paths))
                    
                    # Emit result for display
                    self.result_ready.emit([result_with_filename])
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    
            self.finished_all.emit()
                
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            self.finished_all.emit()

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Image Classification App")
        self.setGeometry(100, 100, 1000, 800)
        
        self.init_ui()
        
        # Store classification results for export
        self.all_results = []
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create a tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.single_image_tab = QWidget()
        self.batch_process_tab = QWidget()
        
        self.tab_widget.addTab(self.single_image_tab, "Single Image")
        self.tab_widget.addTab(self.batch_process_tab, "Batch Processing")
        
        # Setup each tab
        self.setup_single_image_tab()
        self.setup_batch_process_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Add model selection area (common to both tabs)
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        for model in MODELS:
            self.model_selector.addItem(model)
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_selector)
        
        # Download model button
        self.download_model_btn = QPushButton("Download Selected Model to Local")
        self.download_model_btn.clicked.connect(self.download_model)
        model_layout.addWidget(self.download_model_btn)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def setup_single_image_tab(self):
        layout = QVBoxLayout()
        
        # Image selection
        img_select_layout = QHBoxLayout()
        img_select_layout.addWidget(QLabel("Image:"))
        self.image_path_label = QLabel("No image selected")
        img_select_layout.addWidget(self.image_path_label)
        self.browse_img_btn = QPushButton("Browse...")
        self.browse_img_btn.clicked.connect(self.select_single_image)
        img_select_layout.addWidget(self.browse_img_btn)
        layout.addLayout(img_select_layout)
        
        # Image preview
        self.image_preview = QLabel("Image Preview")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(300, 300)
        self.image_preview.setMaximumSize(500, 500)
        self.image_preview.setStyleSheet("border: 1px solid #cccccc;")
        layout.addWidget(self.image_preview)
        
        # Classify button
        self.classify_btn = QPushButton("Classify Image")
        self.classify_btn.clicked.connect(self.classify_single_image)
        layout.addWidget(self.classify_btn)
        
        # Results area
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        self.results_area = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_area.setWidget(self.results_widget)
        self.results_area.setWidgetResizable(True)
        results_layout.addWidget(self.results_area)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.single_image_tab.setLayout(layout)
        
    def setup_batch_process_tab(self):
        layout = QVBoxLayout()
        
        # Folder selection
        folder_select_layout = QHBoxLayout()
        folder_select_layout.addWidget(QLabel("Images Folder:"))
        self.folder_path_label = QLabel("No folder selected")
        folder_select_layout.addWidget(self.folder_path_label)
        self.browse_folder_btn = QPushButton("Browse...")
        self.browse_folder_btn.clicked.connect(self.select_image_folder)
        folder_select_layout.addWidget(self.browse_folder_btn)
        layout.addLayout(folder_select_layout)
        
        # Batch process button
        batch_btn_layout = QHBoxLayout()
        self.batch_process_btn = QPushButton("Process All Images")
        self.batch_process_btn.clicked.connect(self.process_image_batch)
        batch_btn_layout.addWidget(self.batch_process_btn)
        
        self.export_excel_btn = QPushButton("Export Results to Excel")
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        self.export_excel_btn.setEnabled(False)
        batch_btn_layout.addWidget(self.export_excel_btn)
        layout.addLayout(batch_btn_layout)
        
        # Progress bar
        self.progress_group = QGroupBox("Batch Progress")
        progress_layout = QVBoxLayout()
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        progress_layout.addWidget(self.batch_progress)
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)
        
        # Batch results area
        batch_results_group = QGroupBox("Batch Results")
        batch_results_layout = QVBoxLayout()
        self.batch_results_area = QScrollArea()
        self.batch_results_widget = QWidget()
        self.batch_results_layout = QGridLayout()
        self.batch_results_widget.setLayout(self.batch_results_layout)
        self.batch_results_area.setWidget(self.batch_results_widget)
        self.batch_results_area.setWidgetResizable(True)
        batch_results_layout.addWidget(self.batch_results_area)
        batch_results_group.setLayout(batch_results_layout)
        layout.addWidget(batch_results_group)
        
        self.batch_process_tab.setLayout(layout)
    
    def select_single_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if image_path:
            self.image_path_label.setText(image_path)
            self.load_image_preview(image_path)
    
    def load_image_preview(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_preview.setPixmap(pixmap)
    
    def select_image_folder(self):
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Image Folder")
        
        if folder_path:
            self.folder_path_label.setText(folder_path)
            
            # Count valid image files
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            ]
            
            self.progress_label.setText(f"Found {len(self.image_files)} image(s)")
    
    def download_model(self):
        model_id = self.model_selector.currentText()
        
        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        
        model_path = os.path.join("models", model_id.replace("/", "_"))
        
        if os.path.exists(model_path):
            QMessageBox.information(self, "Model Download", f"Model {model_id} already exists locally.")
            return
        
        try:
            QMessageBox.information(self, "Model Download", 
                                   f"Downloading model {model_id}. This may take a while depending on model size. The app will notify you when complete.")
            
            # Download model 
            # This will save the model to the specified path
            pipeline("image-classification", model=model_id, cache_dir="models")
            
            QMessageBox.information(self, "Model Download", f"Model {model_id} has been downloaded successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to download model: {str(e)}")
    
    def classify_single_image(self):
        image_path = self.image_path_label.text()
        if image_path == "No image selected":
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        model_id = self.model_selector.currentText()
        
        # Check if we have a local version
        model_path = os.path.join("models", model_id.replace("/", "_"))
        if os.path.exists(model_path):
            model_to_use = model_path
        else:
            model_to_use = model_id
        
        try:
            # Clear previous results
            self.clear_layout(self.results_layout)
            
            # Create the worker thread
            self.worker = ClassificationWorker(model_to_use, [image_path])
            self.worker.result_ready.connect(self.display_classification_results)
            self.worker.finished_all.connect(lambda: self.classify_btn.setEnabled(True))
            
            # Disable the button while processing
            self.classify_btn.setEnabled(False)
            
            # Start the worker
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Classification failed: {str(e)}")
            self.classify_btn.setEnabled(True)
    
    def process_image_batch(self):
        if not hasattr(self, 'image_files') or not self.image_files:
            QMessageBox.warning(self, "Warning", "Please select a folder with images first.")
            return
        
        model_id = self.model_selector.currentText()
        
        # Check if we have a local version
        model_path = os.path.join("models", model_id.replace("/", "_"))
        if os.path.exists(model_path):
            model_to_use = model_path
        else:
            model_to_use = model_id
        
        # Reset results
        self.all_results = []
        self.clear_layout(self.batch_results_layout)
        self.export_excel_btn.setEnabled(False)
        
        try:
            # Set up worker thread
            self.batch_worker = ClassificationWorker(model_to_use, self.image_files)
            self.batch_worker.progress_update.connect(self.update_batch_progress)
            self.batch_worker.result_ready.connect(self.add_batch_result)
            self.batch_worker.finished_all.connect(self.batch_processing_finished)
            
            # Disable button while processing
            self.batch_process_btn.setEnabled(False)
            
            # Start the worker
            self.batch_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch processing failed: {str(e)}")
            self.batch_process_btn.setEnabled(True)
    
    def update_batch_progress(self, current, total):
        self.batch_progress.setMaximum(total)
        self.batch_progress.setValue(current)
        self.progress_label.setText(f"Processing image {current} of {total}")
    
    def add_batch_result(self, results):
        # Add to overall results for Excel export
        self.all_results.extend(results)
        
        # Add to UI
        for idx, result in enumerate(results):
            filename = result['filename']
            classifications = result['classifications']
            
            # Add results to grid
            row = self.batch_results_layout.rowCount()
            
            # Add filename
            self.batch_results_layout.addWidget(QLabel(filename), row, 0)
            
            # Add top classification
            if classifications:
                top_label = classifications[0]['label']
                top_score = classifications[0]['score']
                result_label = QLabel(f"{top_label} ({top_score:.2f})")
                self.batch_results_layout.addWidget(result_label, row, 1)
            
    def batch_processing_finished(self):
        self.batch_process_btn.setEnabled(True)
        self.export_excel_btn.setEnabled(True)
        QMessageBox.information(self, "Batch Processing", "Batch processing completed successfully!")
    
    def export_to_excel(self):
        if not self.all_results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        # Ask user where to save the Excel file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)")
        
        if not file_path:
            return
        
        try:
            # Prepare data for export
            data = []
            
            for result in self.all_results:
                filename = result['filename']
                classifications = result['classifications']
                
                # Create a row for each classification of each image
                for cls in classifications:
                    if cls['score'] >= MIN_ACCEPTABLE_SCORE:
                        data.append({
                            'Filename': filename,
                            'Label': cls['label'],
                            'Score': cls['score']
                        })
            
            # Convert to DataFrame and export
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def display_classification_results(self, results):
        for result in results:
            classifications = result['classifications']
            
            # Clear previous results
            self.clear_layout(self.results_layout)
            
            # Add header
            self.results_layout.addWidget(QLabel("Classification Results:"))
            
            # Display each classification with progress bar
            comulative_discarded_score = 0
            
            for cls in classifications:
                if cls['score'] >= MIN_ACCEPTABLE_SCORE:
                    # Create container for this result
                    result_container = QWidget()
                    result_layout = QVBoxLayout(result_container)
                    
                    # Add label
                    label_text = QLabel(cls['label'])
                    label_text.setStyleSheet("font-weight: bold;")
                    result_layout.addWidget(label_text)
                    
                    # Add progress bar
                    progress = QProgressBar()
                    progress.setMinimum(0)
                    progress.setMaximum(100)
                    progress.setValue(int(cls['score'] * 100))
                    result_layout.addWidget(progress)
                    
                    # Add score text
                    score_text = QLabel(f"{cls['score']:.4f}")
                    result_layout.addWidget(score_text)
                    
                    self.results_layout.addWidget(result_container)
                else:
                    comulative_discarded_score += cls['score']
            
            # Add discarded score information
            if comulative_discarded_score > 0:
                discarded_container = QWidget()
                discarded_layout = QVBoxLayout(discarded_container)
                
                discarded_label = QLabel("Cumulative Discarded Score:")
                discarded_label.setStyleSheet("font-weight: bold;")
                discarded_layout.addWidget(discarded_label)
                
                discarded_progress = QProgressBar()
                discarded_progress.setMinimum(0)
                discarded_progress.setMaximum(100)
                discarded_progress.setValue(int(comulative_discarded_score * 100))
                discarded_layout.addWidget(discarded_progress)
                
                discarded_score = QLabel(f"{comulative_discarded_score:.4f}")
                discarded_layout.addWidget(discarded_score)
                
                self.results_layout.addWidget(discarded_container)
    
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())