import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import os
from PIL import Image
import imagehash
import sqlite3
import hashlib
import threading
import logging
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from datetime import datetime
import configparser
import json
from queue import Queue
from typing import List, Tuple, Optional
import time

class ImageScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scanner v2.0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize core attributes
        self.is_scanning = False
        self.stop_scan = False
        self.batch_queue = Queue()
        self.setup_logging()
        self.load_config()
        self.setup_db_connection()  # This line was failing
        self.setup_image_processor()
        self.setup_gui()

    def setup_logging(self):
        """Configure logging with file and console output"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler('image_scanner.log')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_config(self) -> None:
        """Load configuration with default values and error handling"""
        self.config = configparser.ConfigParser()
        
        # Define default configuration
        default_config = {
            'DATABASE': {
                'db_path': 'image_db.sqlite'
            },
            'PROCESSING': {
                'batch_size': '50',
                'max_workers': '4',
                'cache_size': '1000'
            }
        }
        
        # Try to read existing config, use defaults if it fails
        try:
            if os.path.exists('config.ini'):
                self.config.read('config.ini')
            else:
                self.config.read_dict(default_config)
                with open('config.ini', 'w') as configfile:
                    self.config.write(configfile)
            
            # Ensure all required sections and keys exist
            for section, values in default_config.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                for key, value in values.items():
                    if not self.config.has_option(section, key):
                        self.config.set(section, key, value)
                        
        except Exception as e:
            self.logger.error(f"Config loading failed: {e}")
            # Fallback to default config
            self.config.read_dict(default_config)
            self.logger.info("Using default configuration")

    def setup_db_connection(self) -> None:
        """Setup SQLite database with connection pooling and error handling"""
        try:
            # Get db_path with fallback
            db_path = self.config.get('DATABASE', 'db_path', fallback='image_db.sqlite')
            self.db_path = os.path.abspath(db_path)  # Use absolute path for reliability
            
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None
            )
            self.create_tables()
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.logger.info(f"Database connected at: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
            raise

    def create_tables(self) -> None:
        """Create database tables with improved schema"""
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        format TEXT,
                        width INTEGER,
                        height INTEGER,
                        description TEXT,
                        file_hash TEXT UNIQUE,
                        perceptual_hash TEXT,
                        absolute_path TEXT NOT NULL,
                        relative_path TEXT,
                        image_vector BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_scanned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                """)
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS scan_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        folder_path TEXT,
                        start_time TEXT,
                        end_time TEXT,
                        total_images INTEGER,
                        status TEXT
                    )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON images (file_hash)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON images (perceptual_hash)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON images (absolute_path)")
        except Exception as e:
            self.logger.error(f"Table creation error: {e}")
            raise

    def setup_image_processor(self) -> None:
        """Setup CLIP model with error handling and optimization"""
        try:
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
        except Exception as e:
            self.log_message(f"Image processor setup error: {e}")
            raise

    def setup_gui(self) -> None:
        """Setup enhanced GUI with additional controls"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Scan Directory")
        self.folder_frame.pack(fill=tk.X, pady=5)
        
        self.folder_path = tk.StringVar()
        ttk.Entry(self.folder_frame, textvariable=self.folder_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(self.folder_frame, text="Browse", command=self.browse_folder).pack(side=tk.RIGHT)

        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(fill=tk.X, pady=5)
        
        self.scan_btn = ttk.Button(self.control_frame, text="Start Scan", command=self.start_scan)
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_scanning, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(self.control_frame, text="Idle")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        self.progress_frame = ttk.LabelFrame(self.main_frame, text="Progress")
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X)
        
        self.stats_label = ttk.Label(self.progress_frame, text="Processed: 0/0")
        self.stats_label.pack()

        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=os.path.expanduser("~"))
        if folder:
            self.folder_path.set(folder)

    def log_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(message)
        def update_gui():
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        self.root.after(0, update_gui)

    def get_image_vector(self, image_path: str) -> np.ndarray:
        """Get image feature vector with caching"""
        try:
            with Image.open(image_path) as image:
                inputs = self.processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                return image_features.cpu().numpy()[0]
        except Exception as e:
            self.log_message(f"Error generating vector for {image_path}: {e}")
            return None

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.log_message(f"Error calculating hash for {file_path}: {e}")
            return None

    def process_image(self, file_path: str, relative_path: str) -> Optional[Tuple]:
        """Process single image and return database row"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                phash = str(imagehash.average_hash(img))
            
            file_size = os.path.getsize(file_path)
            file_hash = self.calculate_file_hash(file_path)
            if not file_hash:
                return None
                
            image_vector = self.get_image_vector(file_path)
            if image_vector is None:
                return None
                
            return (
                os.path.basename(file_path), file_size, format_name, width, height, '',
                file_hash, phash, file_path, relative_path, image_vector.tobytes()
            )
        except Exception as e:
            self.log_message(f"Error processing {file_path}: {e}")
            return None

    def batch_processor(self) -> None:
        """Background thread for processing batches"""
        batch_size = int(self.config['PROCESSING']['batch_size'])
        while self.is_scanning or not self.batch_queue.empty():
            batch = []
            while len(batch) < batch_size and not self.batch_queue.empty():
                batch.append(self.batch_queue.get())
            
            if batch:
                try:
                    with self.conn:
                        self.conn.executemany("""
                            INSERT OR IGNORE INTO images (
                                name, size, format, width, height, description,
                                file_hash, perceptual_hash, absolute_path,
                                relative_path, image_vector
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch)
                except Exception as e:
                    self.log_message(f"Batch insert error: {e}")
            time.sleep(0.1)  # Prevent CPU thrashing

    def scan_images(self) -> None:
        """Scan images with improved performance and status tracking"""
        folder_path = self.folder_path.get()
        if not folder_path or not os.path.exists(folder_path):
            self.log_message("Invalid or no folder selected!")
            self.cleanup_scan()
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        total_files = sum(1 for root, _, files in os.walk(folder_path)
                         for f in files if os.path.splitext(f)[1].lower() in image_extensions)
        
        if total_files == 0:
            self.log_message("No images found in selected folder!")
            self.cleanup_scan()
            return

        scan_id = self.log_scan_start(folder_path, total_files)
        processed_files = 0
        
        try:
            for root, _, files in os.walk(folder_path):
                if self.stop_scan:
                    break
                    
                for filename in files:
                    if self.stop_scan:
                        break
                        
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in image_extensions:
                        continue
                        
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, folder_path)
                    
                    result = self.process_image(file_path, relative_path)
                    if result:
                        self.batch_queue.put(result)
                        processed_files += 1
                        self.update_progress(processed_files, total_files)

            self.log_scan_complete(scan_id, processed_files)
            
        except Exception as e:
            self.log_message(f"Scan error: {e}")
        finally:
            self.cleanup_scan()

    def log_scan_start(self, folder_path: str, total_images: int) -> int:
        """Log scan start in history table"""
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO scan_history (folder_path, start_time, total_images, status)
                VALUES (?, ?, ?, ?)
            """, (folder_path, datetime.now().isoformat(), total_images, 'running'))
            return cursor.lastrowid

    def log_scan_complete(self, scan_id: int, processed_files: int) -> None:
        """Update scan history with completion status"""
        with self.conn:
            self.conn.execute("""
                UPDATE scan_history 
                SET end_time = ?, total_images = ?, status = 'completed'
                WHERE id = ?
            """, (datetime.now().isoformat(), processed_files, scan_id))

    def update_progress(self, processed: int, total: int) -> None:
        """Update GUI progress indicators"""
        progress = (processed / total) * 100
        def update_gui():
            self.progress_var.set(progress)
            self.stats_label.config(text=f"Processed: {processed}/{total}")
            self.status_label.config(text=f"Scanning ({processed}/{total})")
        self.root.after(0, update_gui)

    def start_scan(self) -> None:
        """Start the scanning process"""
        if not self.is_scanning:
            self.is_scanning = True
            self.scan_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Starting...")
            self.progress_var.set(0)
            
            # Start batch processor thread
            threading.Thread(target=self.batch_processor, daemon=True).start()
            threading.Thread(target=self.scan_images, daemon=True).start()

    def stop_scanning(self) -> None:
        """Stop the scanning process"""
        self.stop_scan = True
        self.status_label.config(text="Stopping...")
        self.log_message("Stopping scan...")

    def cleanup_scan(self) -> None:
        """Cleanup after scan completion"""
        self.is_scanning = False
        self.stop_scan = False
        def update_gui():
            self.scan_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Idle")
        self.root.after(0, update_gui)

    def on_closing(self) -> None:
        """Handle window closing"""
        self.stop_scan = True
        time.sleep(0.5)  # Give threads time to cleanup
        self.__del__()
        self.root.destroy()

    def __del__(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageScanner(root)
    try:
        root.mainloop()
    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        app.__del__()