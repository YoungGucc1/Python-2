import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import json
import os
import shutil
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import torch

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    torch.cuda.empty_cache()  # Clear CUDA cache before starting
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Initialize the YOLOv8 model with CUDA if available
model = YOLO("yolo11n.pt")
model.to(DEVICE)  # Move model to GPU if available

# Global variables for control
stop_flag = False
pause_flag = False

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.resizable(False, False)
        self.root.title("Object Detection with YOLOv8" + (" (CUDA)" if CUDA_AVAILABLE else " (CPU)"))
        self.root.geometry("1024x768")
        
        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.image_frame = ttk.Frame(root, padding="10")
        self.image_frame.pack(fill=tk.BOTH, expand=False)
        
        self.status_frame = ttk.Frame(root, padding="10")
        self.status_frame.pack(fill=tk.X)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Action.TButton", padding=10, font=('Arial', 10, 'bold'))
        style.configure("Status.TLabel", font=('Arial', 10))
        
        # Display CUDA status
        self.cuda_label = ttk.Label(
            self.control_frame,
            text=f"GPU Acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}",
            style="Status.TLabel"
        )
        self.cuda_label.pack(pady=5)
        
        # Control buttons frame
        self.create_control_buttons()
        
        # Filter buttons frame
        self.create_filter_buttons()
        
        # Progress and status
        self.create_status_section()
        
        # Display image
        self.create_image_display()
        
        # Initialize variables
        self.folder_path = ""
        self.result_data = {}
        self.files = []
        self.total_files = 0
        
        # Batch processing settings
        self.batch_size = 4 if CUDA_AVAILABLE else 1  # Larger batch size for GPU

    def create_control_buttons(self):
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Select folder button
        self.select_btn = ttk.Button(
            buttons_frame,
            text="Select Folder",
            command=self.select_folder,
            style="Action.TButton"
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Start button
        self.start_button = ttk.Button(
            buttons_frame,
            text="Start Detection",
            command=self.start_detection,
            state="disabled",
            style="Action.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Pause button
        self.pause_button = ttk.Button(
            buttons_frame,
            text="Pause",
            command=self.pause_detection,
            state="disabled",
            style="Action.TButton"
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(
            buttons_frame,
            text="Stop",
            command=self.stop_detection,
            state="disabled",
            style="Action.TButton"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def create_filter_buttons(self):
        filter_frame = ttk.Frame(self.control_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        # Draw and Save button
        self.draw_button = ttk.Button(
            filter_frame,
            text="Draw and Save Images",
            command=self.draw_and_save_images,
            state="disabled",
            style="Action.TButton"
        )
        self.draw_button.pack(side=tk.LEFT, padx=5)
        
        # Move Person button
        self.move_person_button = ttk.Button(
            filter_frame,
            text="Move Person Images",
            command=self.move_person_images,
            state="disabled",
            style="Action.TButton"
        )
        self.move_person_button.pack(side=tk.LEFT, padx=5)
        
        # Move Other button
        self.move_other_button = ttk.Button(
            filter_frame,
            text="Move Non-Person Images",
            command=self.move_other_images,
            state="disabled",
            style="Action.TButton"
        )
        self.move_other_button.pack(side=tk.LEFT, padx=5)

    def create_status_section(self):
        # Detection progress
        ttk.Label(self.status_frame, text="Detection Progress:", style="Status.TLabel").pack(fill=tk.X)
        self.detection_progress = ttk.Progressbar(
            self.status_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.detection_progress.pack(fill=tk.X, pady=2)
        
        # Drawing progress
        ttk.Label(self.status_frame, text="Drawing Progress:", style="Status.TLabel").pack(fill=tk.X)
        self.drawing_progress = ttk.Progressbar(
            self.status_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.drawing_progress.pack(fill=tk.X, pady=2)
        
        # Moving progress
        ttk.Label(self.status_frame, text="Moving Progress:", style="Status.TLabel").pack(fill=tk.X)
        self.moving_progress = ttk.Progressbar(
            self.status_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.moving_progress.pack(fill=tk.X, pady=2)
        
        # Status label
        self.file_label = ttk.Label(
            self.status_frame,
            text="No folder selected",
            style="Status.TLabel"
        )
        self.file_label.pack(pady=5)

    def create_image_display(self):
        # Create a frame for the image with a border
        self.image_container = ttk.Frame(
            self.image_frame,
            borderwidth=2,
            relief="solid"
        )
        self.image_container.pack(expand=False, fill=tk.BOTH, pady=10)
        
        self.image_label = ttk.Label(self.image_container)
        self.image_label.pack(expand=False, pady=10)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.start_button.config(state="normal")
            self.file_label.config(text=f"Selected folder: {self.folder_path}")

    def start_detection(self):
        global stop_flag, pause_flag
        stop_flag = False
        pause_flag = False
        
        # Update button states
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.draw_button.config(state="disabled")
        self.move_person_button.config(state="disabled")
        self.move_other_button.config(state="disabled")

        self.files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.total_files = len(self.files)
        
        self.detection_progress["maximum"] = self.total_files
        self.drawing_progress["maximum"] = self.total_files
        self.moving_progress["maximum"] = self.total_files
        
        # Reset progress bars
        self.detection_progress["value"] = 0
        self.drawing_progress["value"] = 0
        self.moving_progress["value"] = 0
        
        self.result_data = {}
        
        # Start detection thread
        self.thread = threading.Thread(target=self.process_images)
        self.thread.start()

    def process_images(self):
        try:
            # Process images in batches for better GPU utilization
            for i in range(0, len(self.files), self.batch_size):
                if stop_flag:
                    break
                
                while pause_flag:
                    if stop_flag:
                        return
                
                batch_files = self.files[i:i + self.batch_size]
                batch_images = []
                batch_paths = []
                
                # Prepare batch
                for filename in batch_files:
                    file_path = os.path.join(self.folder_path, filename)
                    image = cv2.imread(file_path)
                    batch_images.append(image)
                    batch_paths.append(file_path)
                
                # Process batch
                with torch.cuda.amp.autocast() if CUDA_AVAILABLE else torch.no_grad():
                    results = model(batch_images, device=DEVICE)
                
                # Process results
                for idx, (filename, result) in enumerate(zip(batch_files, results)):
                    detections = []
                    for r in result.boxes:
                        class_id = int(r.cls.item())
                        confidence = float(r.conf.item())
                        box = [float(x) for x in r.xyxy[0].tolist()]
                        detections.append({
                            "class": class_id,
                            "class_name": model.names[class_id],
                            "confidence": confidence,
                            "box": box
                        })
                    self.result_data[filename] = detections
                    
                    # Update GUI for the last processed image
                    self.display_image(batch_paths[idx])
                    current_count = i + idx + 1
                    self.file_label.config(text=f"Processing: {filename} ({current_count}/{self.total_files})")
                    self.detection_progress["value"] = current_count
                    self.root.update()
            
            self.save_results()
            self.enable_post_detection_buttons()
            messagebox.showinfo("Completed", "Object detection completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            self.reset_buttons()
        finally:
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()

    def enable_post_detection_buttons(self):
        self.draw_button.config(state="normal")
        self.move_person_button.config(state="normal")
        self.move_other_button.config(state="normal")
        self.reset_buttons()

    def move_person_images(self):
        persons_folder = os.path.join(self.folder_path, "Persons")
        os.makedirs(persons_folder, exist_ok=True)
        moved_count = 0
        self.moving_progress["value"] = 0
        total_files = len(self.result_data)

        for idx, (filename, detections) in enumerate(self.result_data.items()):
            has_person = any(d["class_name"].lower() == "person" for d in detections)
            if has_person:
                source = os.path.join(self.folder_path, filename)
                destination = os.path.join(persons_folder, filename)
                shutil.copy2(source, destination)
                moved_count += 1
            
            self.moving_progress["value"] = idx + 1
            self.file_label.config(text=f"Moving person images: {idx + 1}/{total_files}")
            self.root.update()

        messagebox.showinfo("Move Complete", f"Moved {moved_count} images containing persons to {persons_folder}")
        self.file_label.config(text="Move operation completed")

    def move_other_images(self):
        others_folder = os.path.join(self.folder_path, "Not Detected")
        os.makedirs(others_folder, exist_ok=True)
        moved_count = 0
        self.moving_progress["value"] = 0
        total_files = len(self.result_data)

        for idx, (filename, detections) in enumerate(self.result_data.items()):
            has_person = any(d["class_name"].lower() == "person" for d in detections)
            if not has_person:
                source = os.path.join(self.folder_path, filename)
                destination = os.path.join(others_folder, filename)
                shutil.copy2(source, destination)
                moved_count += 1
            
            self.moving_progress["value"] = idx + 1
            self.file_label.config(text=f"Moving non-person images: {idx + 1}/{total_files}")
            self.root.update()

        messagebox.showinfo("Move Complete", f"Moved {moved_count} images without persons to {others_folder}")
        self.file_label.config(text="Move operation completed")

    def display_image(self, file_path):
        # Load and resize image while maintaining aspect ratio
        img = Image.open(file_path)
        display_size = (800, 600)  # Maximum display size
        img.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def pause_detection(self):
        global pause_flag
        pause_flag = not pause_flag
        self.pause_button.config(text="Resume" if pause_flag else "Pause")
    
    def stop_detection(self):
        global stop_flag
        stop_flag = True
        self.reset_buttons()
        self.file_label.config(text="Process stopped.")
    
    def save_results(self):
        with open(os.path.join(self.folder_path, "detection_results.json"), "w") as f:
            json.dump(self.result_data, f, indent=4)
    
    def draw_and_save_images(self):
        results_folder = os.path.join(self.folder_path, "results")
        os.makedirs(results_folder, exist_ok=True)
        self.drawing_progress["value"] = 0
        total_files = len(self.result_data)
        
        try:
            for idx, (filename, detections) in enumerate(self.result_data.items()):
                file_path = os.path.join(self.folder_path, filename)
                image = cv2.imread(file_path)
                
                for detection in detections:
                    box = detection["box"]
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    
                    # Draw bounding box
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    color = (0, 255, 0) if class_name.lower() == "person" else (255, 0, 0)
                    cv2.rectangle(image, start_point, end_point, color, 4)
                    
                    # Add label with GPU-optimized size
                    label = f"{class_name}: {confidence:.2f}"
                    label_position = (start_point[0], start_point[1] - 10)
                    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                              1 if CUDA_AVAILABLE else 2, color, 2 if CUDA_AVAILABLE else 3)
                
                result_path = os.path.join(results_folder, filename)
                cv2.imwrite(result_path, image)
                
                self.drawing_progress["value"] = idx + 1
                self.file_label.config(text=f"Drawing and saving images: {idx + 1}/{total_files}")
                self.root.update()
            
            messagebox.showinfo("Images Saved", f"Annotated images saved in {results_folder}")
            self.file_label.config(text="Drawing operation completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while drawing images: {str(e)}")
        finally:
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()

    def reset_buttons(self):
        self.start_button.config(state="normal")
        self.pause_button.config(state="disabled", text="Pause")
        self.stop_button.config(state="disabled")

    # [Previous methods remain the same: create_control_buttons, create_filter_buttons, 
    # create_status_section, create_image_display, select_folder, start_detection, 
    # enable_post_detection_buttons, move_person_images, move_other_images, 
    # display_image, pause_detection, stop_detection, save_results, reset_buttons]

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()