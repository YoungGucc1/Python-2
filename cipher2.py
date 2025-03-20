import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import secrets
import threading
import time

class ModernCryptoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure File Encryptor")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        # Set color scheme
        self.bg_color = "#2c3e50"  # Dark blue-gray
        self.accent_color = "#3498db"  # Bright blue
        self.text_color = "#ecf0f1"  # Light gray
        self.button_color = "#2980b9"  # Darker blue
        self.success_color = "#27ae60"  # Green
        self.error_color = "#e74c3c"  # Red
        
        self.root.configure(bg=self.bg_color)
        
        # Main frame
        self.main_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=20)
        self.main_frame.pack(fill="both", expand=True)
        
        # Title with icon
        self.title_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.title_frame.pack(fill="x", pady=10)
        
        self.lock_icon = "🔒"  # Unicode lock icon
        self.title_label = tk.Label(
            self.title_frame, 
            text=f"{self.lock_icon} Secure File Encryptor {self.lock_icon}", 
            font=("Helvetica", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        self.title_label.pack()
        
        # File selection section
        self.file_frame = tk.LabelFrame(
            self.main_frame, 
            text="File Selection", 
            font=("Helvetica", 12),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        self.file_frame.pack(fill="x", pady=10)
        
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        
        self.file_entry = tk.Entry(
            self.file_frame, 
            textvariable=self.file_path_var,
            width=40,
            bg="#34495e",
            fg=self.text_color,
            readonlybackground="#34495e",
            state="readonly"
        )
        self.file_entry.pack(side=tk.LEFT, padx=5, pady=10, fill="x", expand=True)
        
        self.select_button = tk.Button(
            self.file_frame, 
            text="Browse",
            bg=self.button_color,
            fg=self.text_color,
            activebackground=self.accent_color,
            activeforeground="white",
            padx=10,
            command=self.select_file
        )
        self.select_button.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Password section
        self.password_frame = tk.LabelFrame(
            self.main_frame, 
            text="Password", 
            font=("Helvetica", 12),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        self.password_frame.pack(fill="x", pady=10)
        
        self.password_label = tk.Label(
            self.password_frame, 
            text="Enter Password:",
            bg=self.bg_color,
            fg=self.text_color
        )
        self.password_label.pack(anchor="w", pady=5)
        
        self.password_entry = tk.Entry(
            self.password_frame, 
            show="•", 
            width=30,
            bg="#34495e",
            fg=self.text_color
        )
        self.password_entry.pack(fill="x", pady=5)
        
        self.show_password_var = tk.BooleanVar()
        self.show_password_checkbox = tk.Checkbutton(
            self.password_frame,
            text="Show password",
            variable=self.show_password_var,
            command=self.toggle_password_visibility,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.show_password_checkbox.pack(anchor="w", pady=5)
        
        self.strength_label = tk.Label(
            self.password_frame, 
            text="Password Strength: Not Rated",
            bg=self.bg_color,
            fg=self.text_color
        )
        self.strength_label.pack(anchor="w", pady=5)
        
        self.password_entry.bind("<KeyRelease>", self.check_password_strength)
        
        # Options section
        self.options_frame = tk.LabelFrame(
            self.main_frame, 
            text="Options", 
            font=("Helvetica", 12),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        self.options_frame.pack(fill="x", pady=10)
        
        self.delete_var = tk.BooleanVar()
        self.delete_checkbox = tk.Checkbutton(
            self.options_frame, 
            text="Delete original file after operation",
            variable=self.delete_var,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.delete_checkbox.pack(anchor="w", pady=5)
        
        self.backup_var = tk.BooleanVar()
        self.backup_checkbox = tk.Checkbutton(
            self.options_frame, 
            text="Create backup before operation",
            variable=self.backup_var,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.backup_checkbox.pack(anchor="w", pady=5)
        
        # Action Buttons
        self.buttons_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.buttons_frame.pack(fill="x", pady=10)
        
        self.encrypt_button = tk.Button(
            self.buttons_frame, 
            text="🔒 Encrypt File",
            bg=self.button_color,
            fg=self.text_color,
            activebackground=self.accent_color,
            activeforeground="white",
            padx=20,
            pady=10,
            state="disabled",
            command=self.start_encrypt_thread
        )
        self.encrypt_button.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill="x")
        
        self.decrypt_button = tk.Button(
            self.buttons_frame, 
            text="🔓 Decrypt File",
            bg=self.button_color,
            fg=self.text_color,
            activebackground=self.accent_color,
            activeforeground="white",
            padx=20,
            pady=10,
            state="disabled",
            command=self.start_decrypt_thread
        )
        self.decrypt_button.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill="x")
        
        # Progress bar
        self.progress_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.progress_frame.pack(fill="x", pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=460
        )
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            self.main_frame, 
            text="Ready",
            font=("Helvetica", 10),
            bg=self.bg_color,
            fg=self.text_color
        )
        self.status_label.pack(pady=10, anchor="w")
        
        # Configure ttk style for progress bar
        self.style = ttk.Style()
        self.style.configure("TProgressbar", thickness=20, troughcolor="#34495e", background=self.accent_color)
        
        # Initialize variables
        self.selected_file = None
        self.is_processing = False

    def toggle_password_visibility(self):
        """Toggle password visibility"""
        if self.show_password_var.get():
            self.password_entry.config(show="")
        else:
            self.password_entry.config(show="•")

    def check_password_strength(self, event=None):
        """Check password strength and update indicator"""
        password = self.password_entry.get()
        if not password:
            self.strength_label.config(text="Password Strength: Not Rated")
            return
            
        # Basic password strength check
        strength = 0
        if len(password) >= 8:
            strength += 1
        if any(c.isdigit() for c in password):
            strength += 1
        if any(c.isupper() for c in password):
            strength += 1
        if any(c.islower() for c in password):
            strength += 1
        if any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
            strength += 1
            
        # Update strength label
        if strength == 0:
            self.strength_label.config(text="Password Strength: Very Weak", fg=self.error_color)
        elif strength == 1:
            self.strength_label.config(text="Password Strength: Weak", fg=self.error_color)
        elif strength == 2:
            self.strength_label.config(text="Password Strength: Moderate", fg="#f39c12")  # Orange
        elif strength == 3:
            self.strength_label.config(text="Password Strength: Good", fg="#f1c40f")  # Yellow
        elif strength == 4:
            self.strength_label.config(text="Password Strength: Strong", fg="#2ecc71")  # Light green
        else:
            self.strength_label.config(text="Password Strength: Very Strong", fg=self.success_color)

    def generate_key(self, password, salt=None):
        """Generate encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=150000,  # Increased iterations for better security
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt

    def select_file(self):
        """Select file to encrypt/decrypt"""
        self.selected_file = filedialog.askopenfilename()
        if self.selected_file:
            self.file_path_var.set(os.path.basename(self.selected_file))
            self.encrypt_button.config(state="normal")
            self.decrypt_button.config(state="normal")
            self.status_label.config(text=f"Selected: {os.path.basename(self.selected_file)}")
        else:
            self.file_path_var.set("No file selected")
            self.encrypt_button.config(state="disabled")
            self.decrypt_button.config(state="disabled")
            self.status_label.config(text="Ready")

    def start_encrypt_thread(self):
        """Start encryption in a separate thread to avoid freezing the UI"""
        if self.is_processing:
            return
            
        thread = threading.Thread(target=self.encrypt_file)
        thread.daemon = True
        thread.start()

    def start_decrypt_thread(self):
        """Start decryption in a separate thread to avoid freezing the UI"""
        if self.is_processing:
            return
            
        thread = threading.Thread(target=self.decrypt_file)
        thread.daemon = True
        thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def create_backup(self, file_path):
        """Create a backup of the file"""
        try:
            backup_path = file_path + ".backup"
            with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            return backup_path
        except Exception as e:
            messagebox.showerror("Backup Error", f"Failed to create backup: {str(e)}")
            return None

    def encrypt_file(self):
        """Encrypt the selected file"""
        if not self.selected_file or not self.password_entry.get():
            messagebox.showerror("Error", "Please select a file and enter a password")
            return

        self.is_processing = True
        try:
            # Update UI
            self.status_label.config(text="Encrypting file...", fg=self.accent_color)
            self.encrypt_button.config(state="disabled")
            self.decrypt_button.config(state="disabled")
            self.select_button.config(state="disabled")
            self.update_progress(10)
            
            # Create backup if option is selected
            if self.backup_var.get():
                self.status_label.config(text="Creating backup...", fg=self.accent_color)
                backup_path = self.create_backup(self.selected_file)
                if not backup_path:
                    raise Exception("Failed to create backup")
                self.update_progress(20)
            
            # Read the original file
            self.status_label.config(text="Reading file...", fg=self.accent_color)
            with open(self.selected_file, 'rb') as f:
                original_data = f.read()
            self.update_progress(40)

            # Get original filename and encode it
            original_filename = os.path.basename(self.selected_file).encode()
            
            # Generate key and encrypt
            self.status_label.config(text="Generating encryption key...", fg=self.accent_color)
            password = self.password_entry.get()
            key, salt = self.generate_key(password)
            fernet = Fernet(key)
            self.update_progress(60)
            
            # Simulate encryption progress
            self.status_label.config(text="Encrypting data...", fg=self.accent_color)
            time.sleep(0.5)  # Simulate work
            
            # Combine filename and data with delimiter
            data_to_encrypt = original_filename + b'|||SEP|||' + original_data
            encrypted_data = fernet.encrypt(data_to_encrypt)
            self.update_progress(80)

            # Generate random hash filename
            random_hash = secrets.token_hex(16)
            output_dir = os.path.dirname(self.selected_file)
            output_file = os.path.join(output_dir, f"{random_hash}.encrypted")

            # Save encrypted file with salt prepended
            self.status_label.config(text="Saving encrypted file...", fg=self.accent_color)
            with open(output_file, 'wb') as f:
                f.write(salt + encrypted_data)
            self.update_progress(90)

            # Delete original file if checkbox is checked
            if self.delete_var.get():
                self.status_label.config(text="Deleting original file...", fg=self.accent_color)
                os.remove(self.selected_file)
                delete_msg = "Original file deleted. "
            else:
                delete_msg = ""
            self.update_progress(100)

            self.status_label.config(text="✓ File encrypted successfully!", fg=self.success_color)
            messagebox.showinfo("Success", f"{delete_msg}Encrypted file saved as: {os.path.basename(output_file)}")
            
        except Exception as e:
            self.status_label.config(text=f"✗ Error: {str(e)}", fg=self.error_color)
            messagebox.showerror("Error", f"Encryption failed: {str(e)}")
        finally:
            # Reset UI
            self.encrypt_button.config(state="normal")
            self.decrypt_button.config(state="normal")
            self.select_button.config(state="normal")
            self.is_processing = False
            # Reset progress bar after 2 seconds
            self.root.after(2000, lambda: self.update_progress(0))

    def decrypt_file(self):
        """Decrypt the selected file"""
        if not self.selected_file or not self.password_entry.get():
            messagebox.showerror("Error", "Please select a file and enter a password")
            return

        self.is_processing = True
        try:
            # Update UI
            self.status_label.config(text="Decrypting file...", fg=self.accent_color)
            self.encrypt_button.config(state="disabled")
            self.decrypt_button.config(state="disabled")
            self.select_button.config(state="disabled")
            self.update_progress(10)
            
            # Create backup if option is selected
            if self.backup_var.get():
                self.status_label.config(text="Creating backup...", fg=self.accent_color)
                backup_path = self.create_backup(self.selected_file)
                if not backup_path:
                    raise Exception("Failed to create backup")
                self.update_progress(20)
            
            # Read the encrypted file
            self.status_label.config(text="Reading encrypted file...", fg=self.accent_color)
            with open(self.selected_file, 'rb') as f:
                file_data = f.read()
            self.update_progress(40)

            # Extract salt and encrypted data
            salt = file_data[:16]
            encrypted_data = file_data[16:]

            # Generate key and decrypt
            self.status_label.config(text="Generating decryption key...", fg=self.accent_color)
            password = self.password_entry.get()
            key, _ = self.generate_key(password, salt)
            fernet = Fernet(key)
            self.update_progress(60)
            
            # Simulate decryption progress
            self.status_label.config(text="Decrypting data...", fg=self.accent_color)
            time.sleep(0.5)  # Simulate work
            
            try:
                decrypted_data = fernet.decrypt(encrypted_data)
            except InvalidToken:
                raise Exception("Invalid password or corrupted file")
            
            self.update_progress(80)

            # Split filename and content
            try:
                filename, content = decrypted_data.split(b'|||SEP|||')
                output_dir = os.path.dirname(self.selected_file)
                output_file = os.path.join(output_dir, filename.decode())
            except ValueError:
                raise Exception("File format is not valid")
            
            # Save decrypted file with original name
            self.status_label.config(text="Saving decrypted file...", fg=self.accent_color)
            with open(output_file, 'wb') as f:
                f.write(content)
            self.update_progress(90)
            
            # Delete encrypted file if checkbox is checked
            if self.delete_var.get():
                self.status_label.config(text="Deleting encrypted file...", fg=self.accent_color)
                os.remove(self.selected_file)
                delete_msg = "Encrypted file deleted. "
            else:
                delete_msg = ""
            self.update_progress(100)

            self.status_label.config(text="✓ File decrypted successfully!", fg=self.success_color)
            messagebox.showinfo("Success", f"{delete_msg}Decrypted file saved as: {filename.decode()}")
            
        except Exception as e:
            self.status_label.config(text=f"✗ Error: {str(e)}", fg=self.error_color)
            messagebox.showerror("Error", f"Decryption failed: {str(e)}")
        finally:
            # Reset UI
            self.encrypt_button.config(state="normal")
            self.decrypt_button.config(state="normal")
            self.select_button.config(state="normal")
            self.is_processing = False
            # Reset progress bar after 2 seconds
            self.root.after(2000, lambda: self.update_progress(0))

def main():
    root = tk.Tk()
    app = ModernCryptoApp(root)
    
    # Set icon if possible (this is system dependent)
    try:
        root.iconbitmap("lock_icon.ico")  # You'll need to create this icon file
    except:
        pass  # Skip if no icon file
        
    root.mainloop()

if __name__ == "__main__":
    main()