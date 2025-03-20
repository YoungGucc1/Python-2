"""
Scanner Controller
Handles the logic and interaction between the model and view.
"""
import os
import time
import threading
from queue import Queue
from typing import List, Tuple, Dict, Optional, Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread


class ScanWorker(QThread):
    """Worker thread for scanning images"""
    # Signals for communication with the controller
    progress_updated = pyqtSignal(int, int, float)
    similar_found = pyqtSignal(str, list)
    scan_complete = pyqtSignal(int, int)
    log_message = pyqtSignal(str, str)
    
    def __init__(self, model, folder_path):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.batch_queue = Queue()
        self.batch_size = int(self.model.config.get('PROCESSING', 'batch_size', fallback='50'))
    
    def run(self):
        """Run the scan process"""
        # Start scan in model
        total_files = self.model.start_scan(self.folder_path)
        
        if total_files == 0:
            self.log_message.emit("No images found in selected folder!", "warning")
            self.scan_complete.emit(0, 0)
            return
        
        # Setup batch processor thread
        batch_thread = threading.Thread(target=self.batch_processor, daemon=True)
        batch_thread.start()
        
        # Start processing images
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        processed_files = 0
        similar_count = 0
        start_time = time.time()
        last_update_time = start_time
        
        try:
            for root, _, files in os.walk(self.folder_path):
                if self.model.stop_scan:
                    break
                    
                for filename in files:
                    if self.model.stop_scan:
                        break
                        
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in image_extensions:
                        continue
                        
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, self.folder_path)
                    
                    # Process image and find similar ones
                    result, similar_images = self.model.process_image(file_path, relative_path)
                    if result:
                        self.batch_queue.put((result, similar_images))
                        processed_files += 1
                        
                        # Update similar images view if any found
                        if similar_images:
                            similar_count += len(similar_images)
                            self.similar_found.emit(file_path, similar_images)
                        
                        # Update progress
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            images_per_second = processed_files / elapsed
                            
                            # Update stats every second
                            if current_time - last_update_time >= 1.0:
                                self.progress_updated.emit(processed_files, total_files, images_per_second)
                                last_update_time = current_time
            
            # Log completion
            self.model.log_scan_complete(self.model.scan_id, processed_files)
            
            # Final stats update
            elapsed = time.time() - start_time
            images_per_second = processed_files / elapsed if elapsed > 0 else 0
            self.progress_updated.emit(processed_files, total_files, images_per_second)
            
            self.log_message.emit(f"Scan completed. Processed {processed_files} images in {elapsed:.1f} seconds.", "info")
            self.log_message.emit(f"Found {similar_count} similar images.", "info")
            
            # Signal completion
            self.scan_complete.emit(processed_files, similar_count)
            
        except Exception as e:
            self.log_message.emit(f"Scan error: {e}", "error")
            self.scan_complete.emit(processed_files, similar_count)
    
    def batch_processor(self) -> None:
        """Process batches of images"""
        while self.model.is_scanning or not self.batch_queue.empty():
            batch = []
            while len(batch) < self.batch_size and not self.batch_queue.empty():
                item = self.batch_queue.get()
                if item and item[0] is not None:  # Only add valid items to the batch
                    batch.append(item)
            
            if batch:
                self.model.insert_image_batch(batch)
            
            time.sleep(0.1)  # Prevent CPU thrashing


class ScannerController(QObject):
    """Controller class for the Image Scanner application"""
    
    # Signals for communicating with the view
    progress_updated = pyqtSignal(int, int, float)
    similar_found = pyqtSignal(str, list)
    scan_complete = pyqtSignal()
    log_message = pyqtSignal(str, str)
    
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        
        # Connect signals from view
        self.connect_signals()
        
        # Initialize view with model data
        self.init_view()
        
        # Current scan worker
        self.scan_worker = None
    
    def connect_signals(self):
        """Connect signals between view and controller"""
        # Connect view signals to controller slots
        self.view.start_scan_requested.connect(self.start_scan)
        self.view.stop_scan_requested.connect(self.stop_scan)
        self.view.delete_image_requested.connect(self.delete_image)
        self.view.open_image_requested.connect(self.open_image)
        self.view.similarity_threshold_changed.connect(self.update_similarity_threshold)
        
        # Connect controller signals to view slots
        self.progress_updated.connect(self.view.update_progress)
        self.similar_found.connect(self.view.add_similar_images)
        self.scan_complete.connect(self.view.scan_completed)
        self.log_message.connect(self.view.log_message)
    
    def init_view(self):
        """Initialize view with model data"""
        # Set initial threshold value
        threshold = float(self.model.config.get('PROCESSING', 'similarity_threshold', fallback='0.85'))
        self.view.set_similarity_threshold(threshold)
        
        # Set database path
        self.view.set_database_path(self.model.db_path)
    
    @pyqtSlot(str)
    def start_scan(self, folder_path):
        """Start the scan process"""
        if not folder_path or not os.path.exists(folder_path):
            self.log_message.emit("Invalid or no folder selected!", "error")
            return
        
        # Clear previous similar images
        self.view.clear_similar_images()
        
        # Create scan worker thread
        self.scan_worker = ScanWorker(self.model, folder_path)
        
        # Connect signals
        self.scan_worker.progress_updated.connect(self.on_progress_updated)
        self.scan_worker.similar_found.connect(self.on_similar_found)
        self.scan_worker.scan_complete.connect(self.on_scan_complete)
        self.scan_worker.log_message.connect(self.on_log_message)
        
        # Start the worker
        self.scan_worker.start()
        
        # Log start
        self.log_message.emit("Scan started...", "info")
    
    @pyqtSlot()
    def stop_scan(self):
        """Stop the scan process"""
        if self.model.is_scanning:
            self.model.stop_scan = True
            self.log_message.emit("Stopping scan...", "warning")
    
    @pyqtSlot(str)
    def delete_image(self, image_path):
        """Delete an image"""
        if self.model.delete_image(image_path):
            self.log_message.emit(f"Deleted: {os.path.basename(image_path)}", "warning")
    
    @pyqtSlot(str)
    def open_image(self, image_path):
        """Open an image using the default viewer"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(image_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', image_path))
            else:  # Linux
                subprocess.call(('xdg-open', image_path))
        except Exception as e:
            self.log_message.emit(f"Error opening image: {e}", "error")
    
    @pyqtSlot(float)
    def update_similarity_threshold(self, value):
        """Update similarity threshold in model"""
        self.model.config.set('PROCESSING', 'similarity_threshold', str(value))
        self.model.save_config()
    
    @pyqtSlot(int, int, float)
    def on_progress_updated(self, processed, total, speed):
        """Handle progress updates from scan worker"""
        self.progress_updated.emit(processed, total, speed)
    
    @pyqtSlot(str, list)
    def on_similar_found(self, original_path, similar_images):
        """Handle similar images found during scan"""
        self.similar_found.emit(original_path, similar_images)
    
    @pyqtSlot(int, int)
    def on_scan_complete(self, processed, similar_count):
        """Handle scan completion"""
        self.model.is_scanning = False
        self.model.save_config()
        self.scan_complete.emit()
    
    @pyqtSlot(str, str)
    def on_log_message(self, message, level):
        """Handle log messages from scan worker"""
        self.log_message.emit(message, level)