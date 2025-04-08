import sys
import os
import subprocess
import ctypes
import time
import re
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox,
    QProgressBar, QTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QIcon # Optional: for window icon

# --- Constants ---
APP_NAME = "PyQt6 VHDX Creator"
VERSION = "1.0"
DEFAULT_VHDX_SIZE_GB = 10
MIN_VHDX_SIZE_GB = 1
MAX_VHDX_SIZE_GB = 1024 * 4 # Limit to 4TB for sanity

# --- Helper Function ---
def is_admin():
    """Check if the script is running with Administrator privileges on Windows."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_folder_size(folder_path):
    """Calculates the total size of a folder in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except OSError:
        return -1 # Indicate error
    return total_size


# --- Worker Thread for VHDX Operations ---
class VHDXWorker(QObject):
    """
    Handles the VHDX creation, formatting, copying, and mounting
    in a separate thread to avoid freezing the GUI.
    """
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # bool: success, str: message

    def __init__(self, source_folder, output_vhdx, vhdx_size_gb, mount_vhdx):
        super().__init__()
        self.source_folder = Path(source_folder)
        self.output_vhdx = Path(output_vhdx)
        self.vhdx_size_bytes = vhdx_size_gb * 1024**3
        self.mount_vhdx = mount_vhdx
        self.is_cancelled = False
        self.drive_letter = None

    def _run_powershell(self, command):
        """Executes a PowerShell command and returns result."""
        if self.is_cancelled:
            raise RuntimeError("Operation Cancelled")
        try:
            # Use shell=True for convenience with PowerShell syntax
            # Ensure paths are properly quoted if they contain spaces
            # Using -ExecutionPolicy Bypass might be needed in restricted environments
            full_command = f'powershell.exe -ExecutionPolicy Bypass -Command "{command}"'
            # print(f"Executing: {full_command}") # Debugging
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                encoding='utf-8', # Be explicit for PowerShell output
                shell=True, # Needed for powershell.exe execution this way
                check=False # We check returncode manually
            )
            # print(f"stdout:\n{result.stdout}") # Debugging
            # print(f"stderr:\n{result.stderr}") # Debugging

            if result.returncode != 0:
                error_msg = f"PowerShell Error (Return Code {result.returncode}):\n{result.stderr or result.stdout}"
                raise RuntimeError(error_msg)
            return result.stdout.strip()

        except FileNotFoundError:
            raise RuntimeError("Error: powershell.exe not found. Is PowerShell installed and in PATH?")
        except Exception as e:
            raise RuntimeError(f"PowerShell execution failed: {e}")

    def _parse_drive_letter(self, format_output):
        """ Tries to parse the drive letter from Format-Volume output """
        # Format-Volume output often includes the drive letter like: DriveLetter : F
        match = re.search(r"DriveLetter\s+:\s+([A-Z])", format_output, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback: Try Get-Partition -> Get-Volume
        try:
            disk_num_command = f'(Get-VHD -Path \\"{self.output_vhdx}\\").DiskNumber'
            disk_num_output = self._run_powershell(disk_num_command)
            if disk_num_output and disk_num_output.isdigit():
                disk_num = int(disk_num_output)
                volume_command = f'Get-Partition -DiskNumber {disk_num} | Get-Volume | Select-Object -ExpandProperty DriveLetter'
                drive_letter_output = self._run_powershell(volume_command)
                if drive_letter_output and len(drive_letter_output) == 1 and drive_letter_output.isalpha():
                     # Filter out potential null characters or extra whitespace
                    cleaned_letter = ''.join(filter(str.isalpha, drive_letter_output))
                    if len(cleaned_letter) == 1:
                       return cleaned_letter.upper()
        except Exception as e:
            self.status_update.emit(f"Warning: Could not reliably determine drive letter after format: {e}")
        return None


    def _copy_files_with_progress(self, src, dest):
        """Uses Robocopy for reliable copying and attempts progress."""
        self.status_update.emit("Starting file copy process (using Robocopy)...")
        src_str = str(src)
        dest_str = str(dest)

        # Ensure destination drive is ready - small delay
        time.sleep(2)

        # Robocopy command - /E copies subdirs (including empty), /ETA shows estimated time
        # /R:2 /W:5 retries twice with 5 sec waits (adjust as needed)
        # /NP hides percentage progress per file (cleaner), /NFL hides file names, /NDL hides dir names
        # Using /TEE to output to console as well, /LOG+ for appending log
        # We capture output to parse later if needed, but ETA provides some info
        log_file = self.output_vhdx.with_suffix(".robocopy.log")
        # Escape paths for the command line
        command = [
            'robocopy',
            f'"{src_str}"',
            f'"{dest_str}"',
            '/E', '/COPYALL', '/ETA', '/R:2', '/W:5', '/NP', '/NFL', '/NDL',
            '/TEE', f'/LOG+:"{log_file}"'
        ]
        command_str = " ".join(command)
        self.status_update.emit(f"Robocopy command: {command_str[:100]}...") # Log truncated command

        # Use Popen for potentially monitoring progress (though complex to parse real-time)
        # For simplicity here, we'll just run it and update status after.
        # Real-time parsing would involve reading stdout line-by-line in a loop.
        try:
            process = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, encoding='utf-8', errors='ignore')
            stdout, stderr = process.communicate() # Wait for completion

            self.status_update.emit(f"Robocopy Output:\n{stdout[-500:]}\n{stderr}") # Show last bit of output

            # Robocopy return codes: < 8 indicate success (some files might be skipped/extra)
            if process.returncode >= 8:
                 raise RuntimeError(f"Robocopy failed with exit code {process.returncode}. Check log: {log_file}")
            elif process.returncode > 1:
                 self.status_update.emit(f"Robocopy completed with code {process.returncode} (might indicate extra files or skipped files). Check log: {log_file}")
            else:
                 self.status_update.emit("Robocopy completed successfully.")

        except FileNotFoundError:
            raise RuntimeError("Error: robocopy.exe not found. Is it installed and in PATH?")
        except Exception as e:
            raise RuntimeError(f"File copy failed: {e}")


    def run(self):
        """Main execution logic for the worker thread."""
        try:
            if not is_admin():
                raise RuntimeError("Administrator privileges are required to create and mount VHDX files.")

            self.status_update.emit(f"Source: {self.source_folder}")
            self.status_update.emit(f"Output: {self.output_vhdx}")
            self.status_update.emit(f"Size: {self.vhdx_size_bytes / 1024**3:.2f} GB")
            self.status_update.emit(f"Mount after creation: {self.mount_vhdx}")
            self.progress_update.emit(0)

            # --- 1. Check Output Directory ---
            if not self.output_vhdx.parent.exists():
                self.status_update.emit(f"Creating output directory: {self.output_vhdx.parent}")
                self.output_vhdx.parent.mkdir(parents=True, exist_ok=True)

            # --- 2. Create VHDX ---
            self.status_update.emit("Creating VHDX file...")
            # Using -BlockSizeBytes 1MB for potentially better performance with large files
            # Default is Dynamic, specify -Fixed for fixed size
            # Note: PowerShell New-VHD defaults to Dynamic if -Fixed is not present
            # For simplicity, we'll let it be dynamic unless specified otherwise (future feature).
            vhdx_create_command = f'New-VHD -Path "{self.output_vhdx}" -SizeBytes {self.vhdx_size_bytes} -BlockSizeBytes 1MB -Confirm:$false'
            self._run_powershell(vhdx_create_command)
            self.status_update.emit("VHDX file created.")
            self.progress_update.emit(15) # Approx progress

            # --- 3. Mount VHDX ---
            self.status_update.emit("Mounting VHDX...")
            # Use -PassThru to get the mounted disk object easily
            # Escape the path properly for PowerShell
            mount_command = f'$mountResult = Mount-VHD -Path \\"{self.output_vhdx}\\" -Passthru; $mountResult.DiskNumber'
            disk_number_output = self._run_powershell(mount_command)
            if not disk_number_output or not disk_number_output.isdigit():
                 raise RuntimeError(f"Failed to get disk number after mounting. Output: {disk_number_output}")
            disk_number = int(disk_number_output)
            self.status_update.emit(f"VHDX mounted as Disk {disk_number}.")
            self.progress_update.emit(30)

            # --- 4. Initialize & Partition Disk ---
            self.status_update.emit(f"Initializing Disk {disk_number}...")
            # Initialize (GPT is default for > 2TB and modern systems)
            init_command = f'Initialize-Disk -Number {disk_number} -PartitionStyle GPT -Confirm:$false'
            try:
                 self._run_powershell(init_command)
            except RuntimeError as e:
                if "already been initialized" in str(e):
                     self.status_update.emit(f"Disk {disk_number} already initialized (continuing).")
                else:
                    raise e # Re-raise other initialization errors

            self.status_update.emit(f"Creating partition on Disk {disk_number}...")
            # Create partition using max size, get the partition object
            partition_command = f'$partition = New-Partition -DiskNumber {disk_number} -UseMaximumSize -AssignDriveLetter; $partition | Format-Volume -FileSystem NTFS -Confirm:$false -Force'

            self.status_update.emit("Formatting volume (NTFS)... this may take a moment.")
            format_output = self._run_powershell(partition_command)
            self.drive_letter = self._parse_drive_letter(format_output)

            if self.drive_letter:
                 self.status_update.emit(f"Volume formatted successfully as Drive {self.drive_letter}:")
            else:
                 self.status_update.emit("Volume formatted, but could not determine drive letter automatically.")
                 # Attempt to find it again (maybe needed slight delay)
                 time.sleep(2)
                 self.drive_letter = self._parse_drive_letter("") # Use fallback method

            if not self.drive_letter:
                 raise RuntimeError("Failed to create or format partition or determine drive letter.")

            self.progress_update.emit(50)

            # --- 5. Copy Files ---
            mount_point = Path(f"{self.drive_letter}:\\")
            self.status_update.emit(f"Copying files from {self.source_folder} to {mount_point}...")
            # Use Robocopy for better performance and reliability
            self._copy_files_with_progress(self.source_folder, mount_point)
            self.progress_update.emit(90)

            # --- 6. Dismount (Optional) ---
            final_message = f"VHDX created successfully at '{self.output_vhdx}'"
            if self.mount_vhdx:
                final_message += f" and mounted as Drive {self.drive_letter}:"
                self.status_update.emit("Keeping VHDX mounted as requested.")
            else:
                self.status_update.emit("Dismounting VHDX...")
                dismount_command = f'Dismount-VHD -Path \\"{self.output_vhdx}\\"'
                self._run_powershell(dismount_command)
                self.status_update.emit("VHDX dismounted.")
                final_message += ". (Not mounted)"

            self.progress_update.emit(100)
            self.status_update.emit("Operation completed successfully.")
            self.finished.emit(True, final_message)

        except Exception as e:
            self.status_update.emit(f"ERROR: {e}")
             # Attempt cleanup: Dismount if mounted and not requested to stay mounted
            if self.drive_letter and not self.mount_vhdx:
                 try:
                     self.status_update.emit("Attempting to dismount VHDX after error...")
                     dismount_command = f'Dismount-VHD -Path \\"{self.output_vhdx}\\" -ErrorAction SilentlyContinue'
                     self._run_powershell(dismount_command)
                     self.status_update.emit("Dismount attempt finished.")
                 except Exception as cleanup_e:
                     self.status_update.emit(f"Could not dismount VHDX during cleanup: {cleanup_e}")

            self.finished.emit(False, f"Operation failed: {e}")
        finally:
             # Ensure progress bar reaches end on completion or stays put on error
             if self.progress_update: # Check if signal still exists
                current_progress = self.progress_bar.value() if hasattr(self, 'progress_bar') else 0
                if current_progress < 100 and not self.is_cancelled :
                     # Don't force 100 if it failed mid-way
                     pass


# --- Main Application Window ---
class VHDXCreatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.init_ui()
        self.load_stylesheet("dark_theme.qss") # Load the theme

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        # Consider setting a fixed size or minimum size
        self.setMinimumSize(600, 450)
        # self.setWindowIcon(QIcon("path/to/your/icon.ico")) # Optional

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Source Folder
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Source Folder:")
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.source_browse_button = QPushButton("Browse...")
        self.source_browse_button.clicked.connect(self.select_source_folder)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_path_edit)
        source_layout.addWidget(self.source_browse_button)
        main_layout.addLayout(source_layout)

        # Output VHDX
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output VHDX:")
        self.output_path_edit = QLineEdit()
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.clicked.connect(self.select_output_vhdx)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_button)
        main_layout.addLayout(output_layout)

        # Size and Mount Options
        options_layout = QHBoxLayout()
        self.size_label = QLabel("VHDX Size (GB):")
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(MIN_VHDX_SIZE_GB, MAX_VHDX_SIZE_GB)
        self.size_spinbox.setValue(DEFAULT_VHDX_SIZE_GB)
        self.size_spinbox.setToolTip(f"Size of the virtual disk ({MIN_VHDX_SIZE_GB}-{MAX_VHDX_SIZE_GB} GB)")

        self.mount_checkbox = QCheckBox("Mount VHDX after creation")
        self.mount_checkbox.setChecked(True) # Default to mounting

        options_layout.addWidget(self.size_label)
        options_layout.addWidget(self.size_spinbox)
        options_layout.addStretch(1) # Add space
        options_layout.addWidget(self.mount_checkbox)
        main_layout.addLayout(options_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # Status Area
        self.status_label = QLabel("Status:")
        self.status_output = QTextEdit()
        self.status_output.setReadOnly(True)
        self.status_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Optional
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.status_output, 1) # Give status area stretch factor

        # Create Button
        self.create_button = QPushButton("Create VHDX")
        self.create_button.clicked.connect(self.start_creation)
        self.create_button.setIconSize(QSize(16, 16)) # If using icons later
        main_layout.addWidget(self.create_button, 0, Qt.AlignmentFlag.AlignRight)


    def load_stylesheet(self, filename):
        """Loads a QSS file."""
        style_file = Path(filename)
        if style_file.is_file():
            with open(style_file, "r") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"Warning: Stylesheet '{filename}' not found. Using default style.")


    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.source_path_edit.setText(folder)

    def select_output_vhdx(self):
        # Ensure default name has .vhdx extension
        default_path = Path(self.output_path_edit.text() or Path.home() / "MyDisk.vhdx")
        if default_path.suffix.lower() != ".vhdx":
             default_path = default_path.with_suffix(".vhdx")

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save VHDX File",
            str(default_path), # Start directory and filename suggestion
            "VHDX Files (*.vhdx);;All Files (*)"
        )
        if filepath:
            # Ensure the chosen path ends with .vhdx
            if not filepath.lower().endswith(".vhdx"):
                filepath += ".vhdx"
            self.output_path_edit.setText(filepath)

    def update_status(self, message):
        """Appends a message to the status text area."""
        self.status_output.append(message)
        # Optionally auto-scroll
        self.status_output.verticalScrollBar().setValue(self.status_output.verticalScrollBar().maximum())


    def update_progress(self, value):
        """Updates the progress bar."""
        self.progress_bar.setValue(value)

    def set_controls_enabled(self, enabled):
        """Enable or disable input controls."""
        self.source_browse_button.setEnabled(enabled)
        self.output_browse_button.setEnabled(enabled)
        # Don't disable path edits, user might want to copy text
        # self.source_path_edit.setEnabled(enabled)
        # self.output_path_edit.setEnabled(enabled)
        self.size_spinbox.setEnabled(enabled)
        self.mount_checkbox.setEnabled(enabled)
        self.create_button.setEnabled(enabled)


    def creation_finished(self, success, message):
        """Called when the worker thread finishes."""
        self.update_status(f"--- Finished ---")
        self.update_status(message)
        self.set_controls_enabled(True) # Re-enable controls

        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Success", message)
        else:
            self.progress_bar.setValue(0) # Reset progress on failure
             # Ensure last error message is visible
            self.status_output.verticalScrollBar().setValue(self.status_output.verticalScrollBar().maximum())
            QMessageBox.critical(self, "Error", f"VHDX creation failed.\n\n{message}\n\nCheck the status log for details.")

        # Clean up worker and thread
        self.worker_thread.quit()
        self.worker_thread.wait() # Wait for thread to terminate cleanly
        self.worker = None
        self.worker_thread = None


    def start_creation(self):
        """Validates inputs and starts the VHDX creation process."""
        source_folder = self.source_path_edit.text()
        output_vhdx = self.output_path_edit.text()
        vhdx_size_gb = self.size_spinbox.value()
        mount_vhdx = self.mount_checkbox.isChecked()

        # --- Input Validation ---
        if not source_folder or not Path(source_folder).is_dir():
            QMessageBox.warning(self, "Input Error", "Please select a valid source folder.")
            return

        if not output_vhdx:
            QMessageBox.warning(self, "Input Error", "Please specify a valid output VHDX file path.")
            return
        output_path = Path(output_vhdx)
        if not output_path.parent.exists():
             try:
                 # Attempt to create parent dir if doesn't exist
                 output_path.parent.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                 QMessageBox.warning(self, "Input Error", f"Output directory does not exist and cannot be created:\n{output_path.parent}\nError: {e}")
                 return
        # Warn if output file exists
        if output_path.exists():
            reply = QMessageBox.question(self, "File Exists",
                                         f"The file '{output_path.name}' already exists.\nDo you want to overwrite it?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        if vhdx_size_gb < MIN_VHDX_SIZE_GB:
             QMessageBox.warning(self, "Input Error", f"VHDX size must be at least {MIN_VHDX_SIZE_GB} GB.")
             return

        # Check available disk space (optional but recommended)
        if PSUTIL_AVAILABLE:
            try:
                free_space = psutil.disk_usage(str(output_path.parent)).free
                required_space = vhdx_size_gb * (1024**3) # Size of VHDX file itself
                if free_space < required_space * 1.05: # Need space for VHDX + a little overhead
                     QMessageBox.warning(self, "Disk Space Low",
                                        f"Insufficient disk space on drive {output_path.drive}.\n"
                                        f"Required: ~{required_space / 1024**3:.2f} GB\n"
                                        f"Available: {free_space / 1024**3:.2f} GB")
                     return
            except Exception as e:
                 print(f"Could not check disk space: {e}") # Non-critical error

        # Check source folder size against VHDX capacity (warn if source is larger)
        try:
             src_size = get_folder_size(source_folder)
             if src_size < 0:
                  self.update_status("Warning: Could not determine source folder size.")
             elif src_size > vhdx_size_gb * (1024**3) * 0.95: # Compare bytes (allow ~5% FS overhead)
                 reply = QMessageBox.question(self, "Size Warning",
                                         f"The source folder size (~{src_size / 1024**3:.2f} GB) "
                                         f"might exceed the capacity of the VHDX ({vhdx_size_gb} GB) after formatting.\n\n"
                                         f"Continue anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return
        except Exception as e:
            self.update_status(f"Warning: Could not check source folder size: {e}")


        # Check for Admin rights BEFORE starting thread
        if not is_admin():
            QMessageBox.critical(self, "Administrator Privileges Required",
                                 "This application requires Administrator privileges to create and manage VHDX files.\n"
                                 "Please restart the application as an Administrator.")
            return


        # --- Start Worker Thread ---
        self.status_output.clear() # Clear previous logs
        self.progress_bar.setValue(0)
        self.update_status("Starting VHDX creation process...")
        self.set_controls_enabled(False) # Disable controls during operation

        # Create and start the thread
        self.worker_thread = QThread()
        self.worker = VHDXWorker(source_folder, output_vhdx, vhdx_size_gb, mount_vhdx)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals from worker to slots in GUI thread
        self.worker.progress_update.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.creation_finished)
        self.worker_thread.started.connect(self.worker.run) # Execute run() when thread starts

        # Clean up thread when worker finishes (important!)
        self.worker.finished.connect(self.worker_thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater) # Schedule worker for deletion
        # self.worker_thread.finished.connect(self.worker_thread.deleteLater) # Schedule thread for deletion

        self.worker_thread.start()

    def closeEvent(self, event):
        # Optional: Handle closing while worker is running
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(self, "Operation in Progress",
                                         "A VHDX operation is currently running. Are you sure you want to exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker:
                    self.worker.is_cancelled = True # Signal worker to stop (if possible)
                    self.update_status("*** Aborting operation... cleanup may take time ***")
                # Allow closing, but the thread might still run for a bit for cleanup
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Apply Dark Theme (ensure dark_theme.qss is in the same directory)
    try:
         qss_file = Path(__file__).parent / "dark_theme.qss"
         with open(qss_file, "r") as f:
             app.setStyleSheet(f.read())
    except FileNotFoundError:
         print("Warning: dark_theme.qss not found. Using default application style.")
    except Exception as e:
         print(f"Error loading stylesheet: {e}")


    main_window = VHDXCreatorApp()
    main_window.show()

    sys.exit(app.exec())