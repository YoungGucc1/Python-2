import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pycdlib
import time
import logging

# Set up logging to help debug issues
logging.basicConfig(filename="iso_creator.log", level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class ISOCreatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISO Creator")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.source_path = tk.StringVar()
        self.dest_path = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.iso_creating = False
        
        # Create the UI
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#4a7abc", padx=10, pady=10)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text="ISO Image Creator", 
            font=("Helvetica", 16, "bold"),
            fg="white",
            bg="#4a7abc"
        )
        header_label.pack()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Source folder selection
        source_frame = tk.LabelFrame(main_frame, text="Source Folder", bg="#f0f0f0", padx=10, pady=10)
        source_frame.pack(fill=tk.X, pady=10)
        
        source_entry = tk.Entry(source_frame, textvariable=self.source_path, width=50)
        source_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        source_button = tk.Button(
            source_frame, 
            text="Browse", 
            command=self.browse_source,
            bg="#4a7abc",
            fg="white",
            activebackground="#3a5a8c"
        )
        source_button.pack(side=tk.RIGHT, padx=5)
        
        # Destination file selection
        dest_frame = tk.LabelFrame(main_frame, text="Destination ISO File", bg="#f0f0f0", padx=10, pady=10)
        dest_frame.pack(fill=tk.X, pady=10)
        
        dest_entry = tk.Entry(dest_frame, textvariable=self.dest_path, width=50)
        dest_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        dest_button = tk.Button(
            dest_frame, 
            text="Browse", 
            command=self.browse_destination,
            bg="#4a7abc",
            fg="white",
            activebackground="#3a5a8c"
        )
        dest_button.pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        progress_frame = tk.Frame(main_frame, bg="#f0f0f0", pady=10)
        progress_frame.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = tk.Label(
            progress_frame, 
            textvariable=self.status_var,
            bg="#f0f0f0"
        )
        self.status_label.pack(pady=5)
        
        # Create button
        button_frame = tk.Frame(main_frame, bg="#f0f0f0", pady=10)
        button_frame.pack()
        
        self.create_button = tk.Button(
            button_frame, 
            text="Create ISO", 
            command=self.start_iso_creation,
            bg="#4a7abc",
            fg="white",
            activebackground="#3a5a8c",
            width=20,
            height=2,
            font=("Helvetica", 10, "bold")
        )
        self.create_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = tk.Button(
            button_frame, 
            text="Cancel", 
            command=self.cancel_operation,
            bg="#d9534f",
            fg="white",
            activebackground="#c9302c",
            width=20,
            height=2,
            font=("Helvetica", 10, "bold"),
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Style configuration for the progress bar
        style = ttk.Style()
        style.configure("TProgressbar", thickness=20, troughcolor="#f0f0f0", background="#4a7abc")
        
    def browse_source(self):
        folder_path = filedialog.askdirectory(title="Select Source Folder")
        if folder_path:
            self.source_path.set(folder_path)
    
    def browse_destination(self):
        file_path = filedialog.asksaveasfilename(
            title="Save ISO File As",
            defaultextension=".iso",
            filetypes=[("ISO Files", "*.iso"), ("All Files", "*.*")]
        )
        if file_path:
            self.dest_path.set(file_path)
    
    def validate_inputs(self):
        source = self.source_path.get()
        dest = self.dest_path.get()
        
        if not source:
            messagebox.showerror("Error", "Please select a source folder")
            return False
        
        if not os.path.isdir(source):
            messagebox.showerror("Error", "Source folder does not exist")
            return False
        
        if not dest:
            messagebox.showerror("Error", "Please specify a destination ISO file")
            return False
        
        dest_dir = os.path.dirname(dest)
        if dest_dir and not os.path.isdir(dest_dir):
            messagebox.showerror("Error", "Destination directory does not exist")
            return False
        
        if not os.access(dest_dir or '.', os.W_OK):
            messagebox.showerror("Error", "No write permission for destination directory")
            return False
        
        return True
    
    def start_iso_creation(self):
        if not self.validate_inputs():
            return
        
        self.iso_creating = True
        self.progress_var.set(0)
        self.status_var.set("Preparing...")
        self.create_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        
        # Start the creation in a separate thread
        self.iso_thread = threading.Thread(target=self.create_iso)
        self.iso_thread.daemon = True
        self.iso_thread.start()
        
        # Start monitoring the progress
        self.root.after(100, self.monitor_progress)
    
    def cancel_operation(self):
        if self.iso_creating:
            self.iso_creating = False
            self.status_var.set("Cancelling...")
            # The thread will check this flag and exit
    
    def monitor_progress(self):
        if not self.iso_creating:
            self.create_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            return
        
        if self.iso_thread.is_alive():
            self.root.after(100, self.monitor_progress)
        else:
            self.create_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            if "Error" not in self.status_var.get() and "Cancelling" not in self.status_var.get():
                self.progress_var.set(100)
                self.status_var.set("ISO creation completed!")
                messagebox.showinfo("Success", f"ISO file created successfully:\n{self.dest_path.get()}")
    
    def create_iso(self):
        try:
            source_path = self.source_path.get()
            iso_path = self.dest_path.get()
            
            # Get list of files to add
            all_files = []
            total_size = 0
            self.status_var.set("Scanning files...")
            logging.info("Starting file scan...")
            
            for root, dirs, files in os.walk(source_path):
                if not self.iso_creating:
                    raise Exception("Operation cancelled by user during scan")
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, source_path)
                    all_files.append((full_path, rel_path))
                    total_size += os.path.getsize(full_path)
            
            if not all_files:
                raise Exception("No files found in source folder")
            
            # Create ISO
            iso = pycdlib.PyCdlib()
            iso.new(interchange_level=3, joliet=True)
            
            # Add files to ISO
            processed_size = 0
            for idx, (full_path, rel_path) in enumerate(all_files):
                if not self.iso_creating:
                    iso.close()
                    if os.path.exists(iso_path):
                        os.remove(iso_path)  # Clean up partial file
                    raise Exception("Operation cancelled by user")
                
                self.status_var.set(f"Adding: {rel_path}")
                logging.debug(f"Adding file: {rel_path}")
                
                # Create ISO paths (both for ISO9660 and Joliet)
                iso_path_str = '/'.join(rel_path.split(os.sep))
                if not iso_path_str.startswith('/'):
                    iso_path_str = '/' + iso_path_str
                
                # Handle Joliet path limitations
                joliet_path = iso_path_str
                if len(joliet_path) > 64:
                    base, ext = os.path.splitext(joliet_path)
                    joliet_path = base[:60 - len(ext)] + ext
                
                # Read file data
                with open(full_path, 'rb') as fp:
                    file_data = fp.read()
                
                # Add file to ISO
                iso.add_file(
                    file_data,
                    iso_path_str,
                    joliet_path=joliet_path
                )
                
                # Update progress
                processed_size += os.path.getsize(full_path)
                progress = (processed_size / total_size) * 100
                self.progress_var.set(progress)
                
                # Allow GUI to refresh
                time.sleep(0.01)
            
            # Write ISO
            self.status_var.set("Writing ISO file...")
            logging.info("Writing ISO to disk...")
            iso.write(iso_path)
            iso.close()
            
            self.status_var.set("ISO created successfully!")
            logging.info("ISO creation completed successfully")
            
        except Exception as e:
            logging.error(f"Error during ISO creation: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to create ISO: {str(e)}")
            self.iso_creating = False
            self.create_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ISOCreatorApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application startup error: {str(e)}")
        print(f"Failed to start application: {str(e)}")