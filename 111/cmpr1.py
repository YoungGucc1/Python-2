import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
from PIL import Image
import imagehash
import sqlite3
import threading
import logging
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from datetime import datetime
import configparser
from typing import List, Tuple, Optional

class ImageComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparator")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Core attributes
        self.is_comparing = False
        self.stop_compare = False
        self.setup_logging()
        self.load_config()
        self.setup_db_connection()
        self.setup_image_processor()
        self.setup_gui()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler('image_comparator.log')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_config(self):
        self.config = configparser.ConfigParser()
        default_config = {
            'DATABASE': {'db_path': 'image_db.sqlite'},
            'PROCESSING': {'similarity_threshold': '0.9'},
            'UI': {'theme': 'dark', 'accent_color': '#4CAF50'}
        }
        if os.path.exists('config.ini'):
            self.config.read('config.ini')
        else:
            self.config.read_dict(default_config)
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)

    def setup_db_connection(self):
        db_path = self.config.get('DATABASE', 'db_path', fallback='image_db.sqlite')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.logger.info(f"Connected to database: {db_path}")

    def setup_image_processor(self):
        try:
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
        except Exception as e:
            self.log_message(f"Image processor setup error: {e}")
            raise

    def setup_gui(self):
        # Apply a modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        accent_color = self.config.get('UI', 'accent_color', fallback='#4CAF50')
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TLabel', background='#2E2E2E', foreground='#FFFFFF')
        self.style.configure('TButton', background=accent_color, foreground='#FFFFFF')
        self.style.map('TButton', background=[('active', '#45A049')])
        self.style.configure('TEntry', fieldbackground='#424242', foreground='#FFFFFF')
        self.style.configure('TLabelFrame', background='#2E2E2E', foreground='#FFFFFF')
        self.style.configure('Treeview', background='#424242', fieldbackground='#424242', foreground='#FFFFFF')
        self.style.configure('Treeview.Heading', background=accent_color, foreground='#FFFFFF')

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Select Images", padding=5)
        self.file_frame.pack(fill=tk.X, pady=5)
        self.files_var = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.files_var)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(self.file_frame, text="Browse", command=self.browse_files).pack(side=tk.RIGHT, padx=5)

        # Controls
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=5)
        self.control_frame.pack(fill=tk.X, pady=5)
        self.compare_btn = ttk.Button(self.control_frame, text="Compare Images", command=self.start_comparison)
        self.compare_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_comparison, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = ttk.Button(self.control_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn = ttk.Button(self.control_frame, text="Export Results", command=self.export_results)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_label = ttk.Label(self.control_frame, text="Idle")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Results
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding=5)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_tree = ttk.Treeview(self.results_frame, columns=('Image', 'Matches', 'Details'), show='tree headings')
        self.results_tree.heading('#0', text='Image')
        self.results_tree.heading('Image', text='Details')
        self.results_tree.heading('Matches', text='Similarity')
        self.results_tree.column('#0', width=200)
        self.results_tree.column('Image', width=300)
        self.results_tree.column('Matches', width=100)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        self.results_tree.bind('<Double-1>', self.on_tree_double_click)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.webp")]
        )
        if files:
            self.files_var.set("; ".join(files))

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(message)

    def get_image_vector(self, image_path: str) -> Optional[np.ndarray]:
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

    def calculate_phash(self, image_path: str) -> Optional[str]:
        try:
            with Image.open(image_path) as img:
                return str(imagehash.average_hash(img))
        except Exception as e:
            self.log_message(f"Error calculating phash for {image_path}: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_matches(self, image_path: str) -> List[Tuple[str, float, str]]:
        vector = self.get_image_vector(image_path)
        phash = self.calculate_phash(image_path)
        if vector is None or phash is None:
            return []

        threshold = float(self.config.get('PROCESSING', 'similarity_threshold', fallback='0.9'))
        matches = []

        cursor = self.conn.execute("SELECT absolute_path, image_vector, perceptual_hash FROM images")
        for row in cursor:
            db_path, db_vector_bytes, db_phash = row
            db_vector = np.frombuffer(db_vector_bytes, dtype=np.float32)
            
            # Vector similarity
            similarity = self.cosine_similarity(vector, db_vector)
            
            # Perceptual hash similarity (hamming distance)
            phash_diff = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(db_phash)
            phash_similarity = 1 - (phash_diff / 64.0)

            if similarity >= threshold or phash_similarity >= threshold:
                matches.append((db_path, similarity, phash_similarity))

        return matches

    def compare_images(self):
        files = self.files_var.get().split("; ")
        if not files or not any(os.path.exists(f) for f in files):
            self.log_message("No valid files selected!")
            self.cleanup_comparison()
            return

        total_files = len(files)
        processed_files = 0

        self.root.after(0, lambda: self.progress.config(maximum=total_files))
        for file_path in files:
            if self.stop_compare:
                break
            if not os.path.exists(file_path):
                continue

            self.log_message(f"Comparing: {os.path.basename(file_path)}")
            matches = self.find_matches(file_path)
            
            # Add to Treeview
            parent = self.results_tree.insert('', 'end', text=os.path.basename(file_path), values=(file_path, "Matches: " + str(len(matches))))
            if matches:
                for match_path, vec_sim, phash_sim in matches:
                    self.results_tree.insert(parent, 'end', text="Match", 
                                           values=(match_path, f"Vec: {vec_sim:.3f}, Phash: {phash_sim:.3f}"))
            else:
                self.results_tree.insert(parent, 'end', text="No matches", values=('', ''))
            
            processed_files += 1
            self.root.after(0, lambda p=processed_files: self.progress.config(value=p))
            self.root.after(0, lambda p=processed_files, t=total_files: 
                           self.status_label.config(text=f"Comparing ({p}/{t})"))

        self.cleanup_comparison()

    def start_comparison(self):
        if not self.is_comparing:
            self.is_comparing = True
            self.compare_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.DISABLED)
            self.export_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Starting...")
            threading.Thread(target=self.compare_images, daemon=True).start()

    def stop_comparison(self):
        self.stop_compare = True
        self.status_label.config(text="Stopping...")
        self.log_message("Stopping comparison...")

    def cleanup_comparison(self):
        self.is_comparing = False
        self.stop_compare = False
        self.root.after(0, lambda: self.compare_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.clear_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.status_label.config(text="Idle"))
        self.root.after(0, lambda: self.progress.config(value=0))

    def clear_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def export_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        with open(file_path, 'w') as f:
            for item in self.results_tree.get_children():
                image_name = self.results_tree.item(item, 'text')
                f.write(f"Image: {image_name}\n")
                for child in self.results_tree.get_children(item):
                    details, similarity = self.results_tree.item(child)['values']
                    f.write(f"  {self.results_tree.item(child, 'text')}: {details} ({similarity})\n")
                f.write("\n")
        messagebox.showinfo("Export", "Results exported successfully!")

    def on_tree_double_click(self, event):
        item = self.results_tree.selection()
        if not item:
            return
        values = self.results_tree.item(item[0], 'values')
        if values[0] and os.path.exists(values[0]):
            os.startfile(values[0])  # Opens the file with the default application (Windows)

    def on_closing(self):
        self.stop_compare = True
        if hasattr(self, 'conn'):
            self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='#2E2E2E')
    app = ImageComparatorApp(root)
    root.mainloop()