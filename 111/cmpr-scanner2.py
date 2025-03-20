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
from typing import List, Tuple, Optional, Dict
import time
from sklearn.metrics.pairwise import cosine_similarity
import sv_ttk  # Modern theme for tkinter

class ImageScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scanner v3.0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Apply modern theme
        sv_ttk.set_theme("dark")
        
        # Initialize core attributes
        self.is_scanning = False
        self.stop_scan = False
        self.batch_queue = Queue()
        self.setup_logging()
        self.load_config()
        self.setup_db_connection()
        self.setup_image_processor()
        
        # Cache for similar images
        self.vector_cache = {}
        self.max_cache_size = int(self.config.get('PROCESSING', 'cache_size', fallback='1000'))
        
        # GUI
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
                'cache_size': '1000',
                'similarity_threshold': '0.85'
            },
            'UI': {
                'theme': 'dark',
                'log_max_lines': '500'
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
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS similar_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER,
                        similar_to_id INTEGER,
                        similarity_score REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id),
                        FOREIGN KEY (similar_to_id) REFERENCES images (id)
                    )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON images (file_hash)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON images (perceptual_hash)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON images (absolute_path)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_similar_images ON similar_images (image_id, similar_to_id)")
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
        """Setup enhanced GUI with modern controls and similarity view"""
        # Configure root window
        self.root.geometry('900x700')
        self.root.minsize(800, 600)
        
        # Create main container with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Upper section - Controls
        self.control_section = ttk.Frame(self.main_frame)
        self.control_section.pack(fill=tk.X, pady=(0, 10))
        
        # Folder selection frame
        self.folder_frame = ttk.LabelFrame(self.control_section, text="Scan Directory", padding=10)
        self.folder_frame.pack(fill=tk.X, pady=5)
        
        self.folder_path = tk.StringVar()
        ttk.Entry(self.folder_frame, textvariable=self.folder_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(self.folder_frame, text="Browse", command=self.browse_folder, style='Accent.TButton').pack(side=tk.RIGHT)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.control_section)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        self.scan_btn = ttk.Button(self.control_frame, text="Start Scan", command=self.start_scan, style='Accent.TButton')
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_scanning, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Options frame
        self.options_frame = ttk.LabelFrame(self.control_frame, text="Options", padding=5)
        self.options_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.similarity_threshold = tk.DoubleVar(value=float(self.config.get('PROCESSING', 'similarity_threshold', fallback='0.85')))
        ttk.Label(self.options_frame, text="Similarity Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.similarity_slider = ttk.Scale(self.options_frame, from_=0.5, to=1.0, variable=self.similarity_threshold, orient=tk.HORIZONTAL, length=100)
        self.similarity_slider.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.options_frame, textvariable=tk.StringVar(value=lambda: f"{self.similarity_threshold.get():.2f}")).pack(side=tk.LEFT)
        
        # Status indicators
        self.status_frame = ttk.Frame(self.control_frame)
        self.status_frame.pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Idle", font=("", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT)
        
        # Progress frame
        self.progress_frame = ttk.LabelFrame(self.control_section, text="Progress", padding=5)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_frame = ttk.Frame(self.progress_frame)
        self.stats_frame.pack(fill=tk.X, padx=5)
        
        # Statistics labels
        self.stats_processed = ttk.Label(self.stats_frame, text="Processed: 0/0")
        self.stats_processed.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stats_similar = ttk.Label(self.stats_frame, text="Similar found: 0")
        self.stats_similar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stats_speed = ttk.Label(self.stats_frame, text="Speed: 0 img/sec")
        self.stats_speed.pack(side=tk.LEFT)
        
        # Paned window for log and similar images view
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Log section
        self.log_frame = ttk.LabelFrame(self.paned_window, text="Log", padding=5)
        self.paned_window.add(self.log_frame, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Similar images section
        self.similar_frame = ttk.LabelFrame(self.paned_window, text="Similar Images", padding=5)
        self.paned_window.add(self.similar_frame, weight=3)
        
        # Create a treeview for similar images
        self.similar_tree = ttk.Treeview(self.similar_frame, columns=("Original", "Similar", "Score"))
        self.similar_tree.heading("#0", text="ID")
        self.similar_tree.heading("Original", text="Original Image")
        self.similar_tree.heading("Similar", text="Similar Image")
        self.similar_tree.heading("Score", text="Similarity")
        
        self.similar_tree.column("#0", width=50, stretch=False)
        self.similar_tree.column("Score", width=70, stretch=False, anchor=tk.CENTER)
        
        # Add scrollbar to treeview
        similar_scrollbar = ttk.Scrollbar(self.similar_frame, orient=tk.VERTICAL, command=self.similar_tree.yview)
        self.similar_tree.configure(yscrollcommand=similar_scrollbar.set)
        
        similar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.similar_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Context menu for opening images
        self.context_menu = tk.Menu(self.similar_tree, tearoff=0)
        self.context_menu.add_command(label="Open Original", command=self.open_original_image)
        self.context_menu.add_command(label="Open Similar", command=self.open_similar_image)
        
        self.similar_tree.bind("<Button-3>", self.show_context_menu)
        self.similar_tree.bind("<Double-1>", self.on_tree_double_click)
        
        # Set initial pane positions
        self.paned_window.pane(0, weight=1)
        self.paned_window.pane(1, weight=3)
        
        # Status bar
        self.status_bar = ttk.Frame(self.main_frame)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.db_status = ttk.Label(self.status_bar, text=f"Database: {self.db_path}")
        self.db_status.pack(side=tk.LEFT)
        
        self.version_label = ttk.Label(self.status_bar, text="v3.0")
        self.version_label.pack(side=tk.RIGHT)

    def show_context_menu(self, event):
        """Show context menu for similar images treeview"""
        item = self.similar_tree.identify_row(event.y)
        if item:
            self.similar_tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def on_tree_double_click(self, event):
        """Handle double click on similar images treeview"""
        item = self.similar_tree.identify_row(event.y)
        if item:
            self.similar_tree.selection_set(item)
            column = self.similar_tree.identify_column(event.x)
            if column == "#1" or column == "#0":  # Original image column
                self.open_original_image()
            elif column == "#2":  # Similar image column
                self.open_similar_image()

    def open_original_image(self):
        """Open original image from treeview selection"""
        selected = self.similar_tree.selection()
        if selected:
            item_data = self.similar_tree.item(selected[0])
            original_path = item_data['values'][0]
            self.open_image(original_path)

    def open_similar_image(self):
        """Open similar image from treeview selection"""
        selected = self.similar_tree.selection()
        if selected:
            item_data = self.similar_tree.item(selected[0])
            similar_path = item_data['values'][1]
            self.open_image(similar_path)

    def open_image(self, path):
        """Open image using default system viewer"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', path))
            else:  # Linux
                subprocess.call(('xdg-open', path))
        except Exception as e:
            self.log_message(f"Error opening image: {e}")

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=os.path.expanduser("~"))
        if folder:
            self.folder_path.set(folder)

    def log_message(self, message: str, level="info") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if level == "info":
            self.logger.info(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        
        def update_gui():
            # Apply tag based on message level
            tag = level
            self.log_text.tag_configure("info", foreground="white")
            self.log_text.tag_configure("error", foreground="red")
            self.log_text.tag_configure("warning", foreground="orange")
            self.log_text.tag_configure("similar", foreground="cyan")
            
            self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.see(tk.END)
            
            # Limit log size
            max_lines = int(self.config.get('UI', 'log_max_lines', fallback='500'))
            content = self.log_text.get('1.0', tk.END)
            lines = content.count('\n')
            if lines > max_lines:
                self.log_text.delete('1.0', f"{lines - max_lines}.0")
        
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
            self.log_message(f"Error generating vector for {image_path}: {e}", "error")
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
            self.log_message(f"Error calculating hash for {file_path}: {e}", "error")
            return None

    def find_similar_images(self, image_vector: np.ndarray, file_hash: str, file_path: str) -> List[Dict]:
        """Find similar images in the database using vector similarity"""
        similar_images = []
        threshold = self.similarity_threshold.get()
        
        try:
            # Check by file hash first (exact duplicates)
            cursor = self.conn.execute(
                "SELECT id, absolute_path FROM images WHERE file_hash = ? AND absolute_path != ?",
                (file_hash, file_path)
            )
            exact_matches = cursor.fetchall()
            
            for match_id, match_path in exact_matches:
                similar_images.append({
                    "id": match_id,
                    "path": match_path,
                    "score": 1.0,
                    "is_exact": True
                })
            
            # If exact match found, no need to check for similar images
            if similar_images:
                return similar_images
                
            # Load vectors from database for comparison
            cursor = self.conn.execute(
                "SELECT id, absolute_path, image_vector FROM images WHERE file_hash != ? LIMIT 1000",
                (file_hash,)
            )
            
            for db_id, db_path, db_vector_blob in cursor:
                # Skip if path is the same (shouldn't happen with hash check, but just in case)
                if db_path == file_path:
                    continue
                    
                # Convert blob to numpy array
                db_vector = np.frombuffer(db_vector_blob, dtype=np.float32).reshape(1, -1)
                image_vector_reshaped = image_vector.reshape(1, -1)
                
                # Calculate similarity
                sim_score = float(cosine_similarity(image_vector_reshaped, db_vector)[0][0])
                
                if sim_score >= threshold:
                    similar_images.append({
                        "id": db_id,
                        "path": db_path,
                        "score": sim_score,
                        "is_exact": False
                    })
            
            # Sort by similarity score (descending)
            similar_images.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to top 10 results
            return similar_images[:10]
            
        except Exception as e:
            self.log_message(f"Error finding similar images: {e}", "error")
            return []

    def process_image(self, file_path: str, relative_path: str) -> Tuple[Optional[Tuple], List[Dict]]:
        """Process single image and return database row and similar images"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                phash = str(imagehash.average_hash(img))
            
            file_size = os.path.getsize(file_path)
            file_hash = self.calculate_file_hash(file_path)
            if not file_hash:
                return None, []
                
            image_vector = self.get_image_vector(file_path)
            if image_vector is None:
                return None, []
            
            # Find similar images
            similar_images = self.find_similar_images(image_vector, file_hash, file_path)
                
            return (
                (os.path.basename(file_path), file_size, format_name, width, height, '',
                file_hash, phash, file_path, relative_path, image_vector.tobytes()),
                similar_images
            )
        except Exception as e:
            self.log_message(f"Error processing {file_path}: {e}", "error")
            return None, []

    def add_to_similar_tree(self, original_path: str, similar_images: List[Dict]) -> None:
        """Add similar image entries to the treeview"""
        if not similar_images:
            return
            
        def update_gui():
            for idx, img in enumerate(similar_images):
                item_id = f"sim_{self.similar_counter}"
                self.similar_counter += 1
                
                # Format similarity score as percentage
                score_formatted = f"{img['score']*100:.1f}%"
                
                # Add to treeview
                self.similar_tree.insert(
                    "", 
                    "end", 
                    iid=item_id,
                    text=str(item_id), 
                    values=(
                        original_path,
                        img['path'],
                        score_formatted
                    )
                )
                
                # Update similar count
                self.similar_count += 1
                self.stats_similar.config(text=f"Similar found: {self.similar_count}")
                
                # Log similar image found
                if img['is_exact']:
                    message = f"Exact duplicate found: {os.path.basename(original_path)} = {os.path.basename(img['path'])}"
                else:
                    message = f"Similar image found: {os.path.basename(original_path)} ~ {os.path.basename(img['path'])} ({score_formatted})"
                
                self.log_text.tag_configure("similar", foreground="cyan")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, f"{message}\n", "similar")
                self.log_text.see(tk.END)
        
        self.root.after(0, update_gui)

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
                        # Execute many inserts
                        cursor = self.conn.executemany("""
                            INSERT OR IGNORE INTO images (
                                name, size, format, width, height, description,
                                file_hash, perceptual_hash, absolute_path,
                                relative_path, image_vector
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [item[0] for item in batch])
                        
                        # Process similar images
                        for item in batch:
                            image_data, similar_images = item
                            if similar_images:
                                # Get the ID of the inserted image
                                cursor = self.conn.execute(
                                    "SELECT id FROM images WHERE file_hash = ? AND absolute_path = ?",
                                    (image_data[6], image_data[8])
                                )
                                image_id = cursor.fetchone()[0]
                                
                                # Insert similar image relationships
                                for sim in similar_images:
                                    self.conn.execute("""
                                        INSERT OR IGNORE INTO similar_images 
                                        (image_id, similar_to_id, similarity_score) 
                                        VALUES (?, ?, ?)
                                    """, (image_id, sim['id'], sim['score']))
                except Exception as e:
                    self.log_message(f"Batch insert error: {e}", "error")
            time.sleep(0.1)  # Prevent CPU thrashing

    def scan_images(self) -> None:
        """Scan images with improved performance and status tracking"""
        folder_path = self.folder_path.get()
        if not folder_path or not os.path.exists(folder_path):
            self.log_message("Invalid or no folder selected!", "error")
            self.cleanup_scan()
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        total_files = sum(1 for root, _, files in os.walk(folder_path)
                         for f in files if os.path.splitext(f)[1].lower() in image_extensions)
        
        if total_files == 0:
            self.log_message("No images found in selected folder!", "warning")
            self.cleanup_scan()
            return

        # Initialize counters and timers
        scan_id = self.log_scan_start(folder_path, total_files)
        processed_files = 0
        self.similar_count = 0
        self.similar_counter = 0
        start_time = time.time()
        last_update_time = start_time
        
        # Clear previous similar images
        self.similar_tree.delete(*self.similar_tree.get_children())
        
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
                    
                    # Process image and find similar ones
                    result, similar_images = self.process_image(file_path, relative_path)
                    if result:
                        self.batch_queue.put((result, similar_images))
                        processed_files += 1
                        
                        # Update similar images view if any found
                        if similar_images:
                            self.add_to_similar_tree(file_path, similar_images)
                        
                        # Update progress
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            images_per_second = processed_files / elapsed
                            
                            # Update stats every second
                            if current_time - last_update_time >= 1.0:
                                self.update_progress(processed_files, total_files, images_per_second)
                                last_update_time = current_time

            self.log_scan_complete(scan_id, processed_files)
            
            # Final stats update
            elapsed = time.time() - start_time
            images_per_second = processed_files / elapsed if elapsed > 0 else 0
            self.update_progress(processed_files, total_files, images_per_second)
            
            # Log completion
            self.log_message(f"Scan completed. Processed {processed_files} images in {elapsed:.1f} seconds.")
            self.log_message(f"Found {self.similar_count} similar images.")
            
        except Exception as e:
            self.log_message(f"Scan error: {e}", "error")
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
                SET end_time = ?, processed_files = ?, status = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), processed_files, 'completed', scan_id))

    def update_progress(self, processed: int, total: int, speed: float) -> None:
        """Update progress indicators on GUI"""
        def update_gui():
            progress_percent = (processed / total) * 100 if total > 0 else 0
            self.progress_var.set(progress_percent)
            self.stats_processed.config(text=f"Processed: {processed}/{total}")
            self.stats_speed.config(text=f"Speed: {speed:.1f} img/sec")
        
        self.root.after(0, update_gui)

    def start_scan(self) -> None:
        """Start the scanning process"""
        if self.is_scanning:
            return
            
        self.is_scanning = True
        self.stop_scan = False
        
        # Update UI state
        self.scan_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Scanning...")
        
        # Start batch processor thread
        processor_thread = threading.Thread(target=self.batch_processor, daemon=True)
        processor_thread.start()
        
        # Start scanner thread
        scanner_thread = threading.Thread(target=self.scan_images, daemon=True)
        scanner_thread.start()
        
        self.log_message("Scan started...")

    def stop_scanning(self) -> None:
        """Stop the scanning process"""
        self.stop_scan = True
        self.log_message("Stopping scan...")
        self.status_label.config(text="Stopping...")

    def cleanup_scan(self) -> None:
        """Clean up after scan completion"""
        self.is_scanning = False
        self.stop_scan = False
        
        # Update UI state
        self.scan_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Idle")
        
        # Save config
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

    def on_closing(self) -> None:
        """Handle window closing"""
        if self.is_scanning:
            if tk.messagebox.askokcancel("Quit", "A scan is running. Do you want to abort and quit?"):
                self.stop_scan = True
                # Wait a bit for threads to clean up
                time.sleep(1)
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    # Set up main window
    root = tk.Tk()
    app = ImageScanner(root)
    
    # Center window on screen
    window_width = 900
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width - window_width) / 2)
    y_coordinate = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
    
    # Run the application
    root.mainloop()        