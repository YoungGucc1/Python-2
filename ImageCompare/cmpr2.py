import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
from PIL import Image, ImageTk
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
import json

class ImageComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparator Pro")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Core attributes
        self.is_comparing = False
        self.stop_compare = False
        self.thumbnails = {}  # Store thumbnails for display
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
            'PROCESSING': {'similarity_threshold': '0.9', 'thumbnail_size': '100'},
            'UI': {'theme': 'dark', 'accent_color': '#4CAF50', 'secondary_color': '#2196F3'}
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
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS images (
            absolute_path TEXT PRIMARY KEY,
            image_vector BLOB,
            perceptual_hash TEXT,
            last_modified REAL
        )''')
        self.conn.commit()
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
        # Apply modern theme with improved styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        accent_color = self.config.get('UI', 'accent_color', fallback='#4CAF50')
        secondary_color = self.config.get('UI', 'secondary_color', fallback='#2196F3')
        
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TLabel', background='#2E2E2E', foreground='#FFFFFF')
        self.style.configure('TButton', background=accent_color, foreground='#FFFFFF')
        self.style.map('TButton', background=[('active', '#45A049')])
        self.style.configure('TEntry', fieldbackground='#424242', foreground='#FFFFFF')
        self.style.configure('TLabelFrame', background='#2E2E2E', foreground='#FFFFFF')
        self.style.configure('Treeview', background='#424242', fieldbackground='#424242', foreground='#FFFFFF')
        self.style.configure('Treeview.Heading', background=accent_color, foreground='#FFFFFF')
        self.style.configure('Secondary.TButton', background=secondary_color)
        self.style.map('Secondary.TButton', background=[('active', '#1E88E5')])

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

        # Controls frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=5)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Comparison controls
        self.compare_btn = ttk.Button(self.control_frame, text="Compare Images", command=self.start_comparison)
        self.compare_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_comparison, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = ttk.Button(self.control_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn = ttk.Button(self.control_frame, text="Export Results", command=self.export_results)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Threshold control
        self.threshold_var = tk.DoubleVar(value=float(self.config.get('PROCESSING', 'similarity_threshold', fallback='0.9')))
        ttk.Label(self.control_frame, text="Similarity Threshold:").pack(side=tk.LEFT, padx=5)
        ttk.Scale(self.control_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.threshold_var, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.control_frame, textvariable=self.threshold_var).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_label = ttk.Label(self.control_frame, text="Idle")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding=5)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split results into treeview and preview
        self.results_pane = ttk.PanedWindow(self.results_frame, orient=tk.HORIZONTAL)
        self.results_pane.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        self.tree_frame = ttk.Frame(self.results_pane)
        self.results_pane.add(self.tree_frame, weight=2)
        self.results_tree = ttk.Treeview(self.tree_frame, 
                                       columns=('Image', 'Matches', 'Details', 'Similarity'), 
                                       show='tree headings')
        self.results_tree.heading('#0', text='Image')
        self.results_tree.heading('Image', text='Path')
        self.results_tree.heading('Matches', text='Matches')
        self.results_tree.heading('Details', text='Details')
        self.results_tree.heading('Similarity', text='Similarity')
        self.results_tree.column('#0', width=200)
        self.results_tree.column('Image', width=300)
        self.results_tree.column('Matches', width=100)
        self.results_tree.column('Details', width=150)
        self.results_tree.column('Similarity', width=100)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        self.results_tree.bind('<Double-1>', self.on_tree_double_click)
        self.results_tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # Preview frame
        self.preview_frame = ttk.Frame(self.results_pane)
        self.results_pane.add(self.preview_frame, weight=1)
        self.preview_label = ttk.Label(self.preview_frame, text="Image Preview")
        self.preview_label.pack(fill=tk.X)
        self.preview_image = ttk.Label(self.preview_frame)
        self.preview_image.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Image details
        self.details_frame = ttk.LabelFrame(self.preview_frame, text="Details", padding=5)
        self.details_frame.pack(fill=tk.X, pady=5)
        self.details_text = scrolledtext.ScrolledText(self.details_frame, height=5, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X)
        self.details_text.config(state=tk.DISABLED)

        # Quick action buttons
        self.action_frame = ttk.Frame(self.preview_frame)
        self.action_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self.action_frame, text="Open", style='Secondary.TButton', 
                  command=self.open_selected_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.action_frame, text="Copy Path", style='Secondary.TButton', 
                  command=self.copy_image_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.action_frame, text="Compare More", style='Secondary.TButton', 
                  command=self.compare_selected_image).pack(side=tk.LEFT, padx=5)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.webp")]
        )
        if files:
            self.files_var.set("; ".join(files))

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(message)

    def create_thumbnail(self, image_path: str) -> Optional[ImageTk.PhotoImage]:
        try:
            thumbnail_size = int(self.config.get('PROCESSING', 'thumbnail_size', fallback='100'))
            with Image.open(image_path) as img:
                img.thumbnail((thumbnail_size, thumbnail_size))
                return ImageTk.PhotoImage(img)
        except Exception as e:
            self.log_message(f"Error creating thumbnail for {image_path}: {e}")
            return None

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

        threshold = self.threshold_var.get()
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

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def compare_images(self):
        files = self.files_var.get().split("; ")
        if not files or not any(os.path.exists(f) for f in files):
            self.log_message("No valid files selected!")
            messagebox.showerror("Error", "Please select valid image files!")
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
            
            # Add to Treeview with thumbnail
            thumbnail = self.create_thumbnail(file_path)
            if thumbnail:
                self.thumbnails[file_path] = thumbnail
            
            parent = self.results_tree.insert('', 'end', 
                                           text=os.path.basename(file_path), 
                                           values=(file_path, len(matches), "Click to view", f"N/A"),
                                           image=thumbnail)
            
            if matches:
                for match_path, vec_sim, phash_sim in matches:
                    match_thumbnail = self.create_thumbnail(match_path)
                    if match_thumbnail:
                        self.thumbnails[match_path] = match_thumbnail
                    avg_sim = (vec_sim + phash_sim) / 2
                    self.results_tree.insert(parent, 'end', 
                                          text=os.path.basename(match_path),
                                          values=(match_path, "Match", 
                                                 f"Vec: {vec_sim:.3f}, Phash: {phash_sim:.3f}",
                                                 f"{avg_sim:.3f}"),
                                          image=match_thumbnail)
            else:
                self.results_tree.insert(parent, 'end', 
                                      text="No matches found", 
                                      values=('', '', '', ''))
            
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
        self.thumbnails.clear()
        self.preview_image.config(image='')
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.config(state=tk.DISABLED)

    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        if not file_path:
            return
            
        results = []
        for item in self.results_tree.get_children():
            image_data = {
                "image": self.results_tree.item(item, 'text'),
                "path": self.results_tree.item(item, 'values')[0],
                "matches": []
            }
            for child in self.results_tree.get_children(item):
                child_data = self.results_tree.item(child)
                image_data["matches"].append({
                    "image": child_data['text'],
                    "path": child_data['values'][0],
                    "details": child_data['values'][2],
                    "similarity": child_data['values'][3]
                })
            results.append(image_data)

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        messagebox.showinfo("Export", "Results exported successfully!")

    def on_tree_select(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        values = self.results_tree.item(item, 'values')
        if not values[0] or not os.path.exists(values[0]):
            return

        # Update preview
        try:
            img = Image.open(values[0])
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.preview_image.config(image=photo)
            self.preview_image.image = photo  # Keep reference
        except Exception as e:
            self.log_message(f"Error displaying preview: {e}")

        # Update details
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        details = f"Path: {values[0]}\n"
        details += f"Matches: {values[1]}\n"
        details += f"Details: {values[2]}\n"
        details += f"Average Similarity: {values[3]}"
        self.details_text.insert(tk.END, details)
        self.details_text.config(state=tk.DISABLED)

    def on_tree_double_click(self, event):
        self.open_selected_image()

    def open_selected_image(self):
        selection = self.results_tree.selection()
        if not selection:
            return
        values = self.results_tree.item(selection[0], 'values')
        if values[0] and os.path.exists(values[0]):
            os.startfile(values[0])  # Windows-specific, use platform-appropriate command

    def copy_image_path(self):
        selection = self.results_tree.selection()
        if not selection:
            return
        values = self.results_tree.item(selection[0], 'values')
        if values[0]:
            self.root.clipboard_clear()
            self.root.clipboard_append(values[0])
            messagebox.showinfo("Success", "Path copied to clipboard!")

    def compare_selected_image(self):
        selection = self.results_tree.selection()
        if not selection:
            return
        values = self.results_tree.item(selection[0], 'values')
        if values[0] and os.path.exists(values[0]):
            self.files_var.set(values[0])
            self.start_comparison()

    def on_closing(self):
        self.stop_compare = True
        if hasattr(self, 'conn'):
            self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")  # Default window size
    root.configure(bg='#2E2E2E')
    app = ImageComparatorApp(root)
    root.mainloop()