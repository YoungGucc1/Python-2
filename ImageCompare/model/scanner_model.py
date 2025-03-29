"""
Scanner Model
Handles all data-related operations for the image scanner application.
"""
import os
import time
import hashlib
import sqlite3
import logging
import configparser
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ScannerModel:
    """Model class for the Image Scanner application"""
    
    def __init__(self):
        """Initialize the model with configuration and database"""
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        # Setup database
        self.db_path = os.path.abspath(self.config.get('DATABASE', 'db_path', fallback='image_db.sqlite'))
        self.setup_database()
        
        # Setup image processor if available
        self.processor = None
        self.model = None
        if TRANSFORMERS_AVAILABLE:
            self.setup_image_processor()
        
        # Current scan state
        self.is_scanning = False
        self.stop_scan = False
        self.scan_start_time = None
        self.scan_id = None
        
        # Scan statistics
        self.total_files = 0
        self.processed_files = 0
        self.similar_count = 0
        
        # Vector cache for similar images
        self.vector_cache = {}
        self.max_cache_size = int(self.config.get('PROCESSING', 'cache_size', fallback='1000'))
    
    def setup_logging(self) -> None:
        """Configure logging with file and console output"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Add file handler
        fh = logging.FileHandler('image_scanner.log')
        fh.setLevel(logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def load_config(self) -> configparser.ConfigParser:
        """Load configuration file with default values"""
        config = configparser.ConfigParser()
        
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
                config.read('config.ini')
            else:
                config.read_dict(default_config)
                with open('config.ini', 'w') as configfile:
                    config.write(configfile)
            
            # Ensure all required sections and keys exist
            for section, values in default_config.items():
                if not config.has_section(section):
                    config.add_section(section)
                for key, value in values.items():
                    if not config.has_option(section, key):
                        config.set(section, key, value)
                        
        except Exception as e:
            self.logger.error(f"Config loading failed: {e}")
            # Fallback to default config
            config.read_dict(default_config)
            self.logger.info("Using default configuration")
        
        return config
    
    def setup_database(self) -> None:
        """Setup SQLite database with connection pooling and error handling"""
        try:
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
        """Create database tables with schema"""
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
                        processed_files INTEGER DEFAULT 0,
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
            self.logger.info("CLIP model initialized successfully")
        except Exception as e:
            self.logger.error(f"Image processor setup error: {e}")
            self.processor = None
            self.model = None
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA-256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def get_image_vector(self, image_path: str) -> Optional[np.ndarray]:
        """Get image feature vector using CLIP model"""
        if not self.processor or not self.model:
            self.logger.warning("Image processor not available")
            return None
            
        try:
            with Image.open(image_path) as image:
                inputs = self.processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                return image_features.cpu().numpy()[0]
        except Exception as e:
            self.logger.error(f"Error generating vector for {image_path}: {e}")
            return None
    
    def process_image(self, file_path: str, relative_path: str) -> Tuple[Optional[Tuple], List[Dict]]:
        """Process a single image and return database row and similar images"""
        try:
            # Get image data
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                phash = str(imagehash.average_hash(img))
            
            # Calculate file hash and size
            file_size = os.path.getsize(file_path)
            file_hash = self.calculate_file_hash(file_path)
            if not file_hash:
                return None, []
            
            # Get image vector
            image_vector = self.get_image_vector(file_path)
            if image_vector is None:
                # If vector generation fails, insert without vector
                image_vector_bytes = None
            else:
                image_vector_bytes = image_vector.tobytes()
            
            # Find similar images
            similar_images = []
            if image_vector is not None:
                similar_images = self.find_similar_images(image_vector, file_hash, file_path)
            
            # Return image data and similar images
            return (
                (os.path.basename(file_path), file_size, format_name, width, height, '',
                 file_hash, phash, file_path, relative_path, image_vector_bytes),
                similar_images
            )
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None, []
    
    def find_similar_images(self, image_vector: np.ndarray, file_hash: str, file_path: str) -> List[Dict]:
        """Find similar images in the database using vector similarity"""
        similar_images = []
        threshold = float(self.config.get('PROCESSING', 'similarity_threshold', fallback='0.85'))
        
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
                "SELECT id, absolute_path, image_vector FROM images WHERE file_hash != ? AND image_vector IS NOT NULL LIMIT 1000",
                (file_hash,)
            )
            
            for db_id, db_path, db_vector_blob in cursor:
                # Skip if path is the same (shouldn't happen with hash check, but just in case)
                if db_path == file_path:
                    continue
                
                # Skip if blob is None
                if db_vector_blob is None:
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
            self.logger.error(f"Error finding similar images: {e}")
            return []
    
    def start_scan(self, folder_path: str) -> int:
        """Start scan process"""
        self.folder_path = folder_path
        self.is_scanning = True
        self.stop_scan = False
        self.scan_start_time = time.time()
        self.processed_files = 0
        self.similar_count = 0
        
        # Count total files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.total_files = sum(1 for root, _, files in os.walk(folder_path)
                             for f in files if os.path.splitext(f)[1].lower() in image_extensions)
        
        # Log scan start
        self.scan_id = self.log_scan_start(folder_path, self.total_files)
        
        return self.total_files
    
    def stop_scan(self) -> None:
        """Stop the scan process"""
        self.stop_scan = True
        self.is_scanning = False
    
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
    
    def insert_image_batch(self, batch: List[Tuple[Tuple, List[Dict]]]) -> None:
        """Insert a batch of images into the database"""
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
                        try:
                            # Get the ID of the inserted image
                            cursor = self.conn.execute(
                                "SELECT id FROM images WHERE file_hash = ? AND absolute_path = ?",
                                (image_data[6], image_data[8])
                            )
                            result = cursor.fetchone()
                            if result:
                                image_id = result[0]
                                
                                # Insert similar image relationships
                                for sim in similar_images:
                                    if 'id' in sim:  # Make sure the id key exists
                                        self.conn.execute("""
                                            INSERT OR IGNORE INTO similar_images 
                                            (image_id, similar_to_id, similarity_score) 
                                            VALUES (?, ?, ?)
                                        """, (image_id, sim['id'], sim['score']))
                        except Exception as e:
                            self.logger.error(f"Error processing similar image: {e}")
        except Exception as e:
            self.logger.error(f"Batch insert error: {e}")
    
    def get_image_data(self, image_path: str) -> Dict[str, Any]:
        """Get image data from the database"""
        try:
            cursor = self.conn.execute(
                "SELECT id, name, size, format, width, height, description, file_hash, perceptual_hash, absolute_path, relative_path "
                "FROM images WHERE absolute_path = ?", 
                (image_path,)
            )
            result = cursor.fetchone()
            if result:
                return {
                    "id": result[0],
                    "name": result[1],
                    "size": result[2],
                    "format": result[3],
                    "width": result[4],
                    "height": result[5],
                    "description": result[6],
                    "file_hash": result[7],
                    "perceptual_hash": result[8],
                    "absolute_path": result[9],
                    "relative_path": result[10]
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting image data: {e}")
            return None
    
    def delete_image(self, image_path: str) -> bool:
        """Delete an image from the filesystem and database"""
        try:
            # Delete from filesystem
            os.remove(image_path)
            
            # Delete from database
            with self.conn:
                self.conn.execute(
                    "DELETE FROM images WHERE absolute_path = ?", 
                    (image_path,)
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting image: {e}")
            return False
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.conn:
            self.conn.close()
        self.logger.info("Resources cleaned up")