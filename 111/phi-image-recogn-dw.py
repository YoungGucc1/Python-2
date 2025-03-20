import tkinter as tk
from tkinter import filedialog, ttk
import os
import json
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import threading

class ImageRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recognition with Phi-3.5-vision")
        self.root.geometry("800x600")

        # Initialize model variables
        self.model = None
        self.processor = None
        self.results = {}

        # Create GUI elements
        self.create_widgets()

        # Load model in a separate thread
        self.status_label.config(text="Loading model...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Frame for buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Select folder button
        self.select_button = ttk.Button(button_frame, text="Select Folder", command=self.select_folder)
        self.select_button.pack(side=tk.LEFT, padx=5)

        # Process button
        self.process_button = ttk.Button(button_frame, text="Process Images", command=self.process_images, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=5)

        # Save button
        self.save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results, state="disabled")
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(self.root, text="Initializing...")
        self.status_label.pack(pady=5)

        # Results text area
        self.results_text = tk.Text(self.root, height=25, width=90)
        self.results_text.pack(pady=10)

        # Scrollbar for text area
        scrollbar = ttk.Scrollbar(self.root, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def load_model(self):
        try:
            # Load the Phi-3.5-vision-instruct model and processor with Flash Attention
            model_id = "microsoft/Phi-3.5-vision-instruct"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    _attn_implementation="flash_attention_2"
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                self.status_label.config(text="Model loaded successfully with Flash Attention!")
            except Exception as flash_error:
                self.status_label.config(text=f"Flash Attention failed: {str(flash_error)}. Falling back to eager.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    _attn_implementation="eager"
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                self.status_label.config(text="Model loaded successfully with eager attention!")
           
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.process_button.config(state="normal")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            self.status_label.config(text=f"Selected folder: {folder_path}")
            self.results_text.delete(1.0, tk.END)
            self.results = {}
            self.save_button.config(state="disabled")

    def process_images(self):
        if not hasattr(self, 'folder_path') or not self.model or not self.processor:
            self.status_label.config(text="Please select a folder and wait for model to load!")
            return

        self.status_label.config(text="Processing images...")
        self.process_button.config(state="disabled")
        self.results_text.delete(1.0, tk.END)
        self.results = {}

        threading.Thread(target=self.process_images_thread, daemon=True).start()

    
    
    def save_results(self):
        if not self.results:
            self.status_label.config(text="No results to save!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                self.status_label.config(text=f"Results saved to {save_path}")
            except Exception as e:
                self.status_label.config(text=f"Error saving results: {str(e)}")

def main():
    root = tk.Tk()
    app = ImageRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()