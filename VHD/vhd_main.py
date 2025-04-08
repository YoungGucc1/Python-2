import sys
import os
import subprocess
import ctypes
import time
import re
import shutil
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox,
    QProgressBar, QTextEdit, QMessageBox, QSpinBox, QTabWidget, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QIcon, QPalette, QColor # Added QPalette, QColor for basic fallback style

# --- Constants ---
APP_NAME = "PyQt6 VHDX/ISO Creator"
VERSION = "1.1" # Incremented version
DEFAULT_VHDX_SIZE_GB = 10
MIN_VHDX_SIZE_GB = 1
MAX_VHDX_SIZE_GB = 1024 * 4 # Limit to 4TB for sanity

# --- Dark Theme Stylesheet (as a multi-line string) ---
DARK_THEME_QSS = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    border: none;
    font-size: 10pt; /* Adjust font size as needed */
}

QMainWindow {
    background-color: #3c3f41; /* Slightly different for main window */
}

QGroupBox {
    background-color: #383838; /* Slightly different background for groups */
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 10px; /* Space for the title */
    padding: 10px 5px 5px 5px; /* Padding inside the box */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    margin-left: 5px;
    background-color: #383838; /* Match group background */
    color: #e0e0e0;
}


QPushButton {
    background-color: #555555;
    color: #f0f0f0;
    border: 1px solid #666666;
    padding: 5px 10px;
    border-radius: 4px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #666666;
    border: 1px solid #777777;
}

QPushButton:pressed {
    background-color: #444444;
}

QPushButton:disabled {
    background-color: #404040;
    color: #808080;
    border-color: #505050;
}


QLineEdit, QTextEdit, QSpinBox {
    background-color: #3c3f41;
    color: #f0f0f0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}

QLineEdit:read-only {
    background-color: #333333;
}


QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    background-color: #555555;
    border: none; /* Border is part of the main spinbox */
    width: 16px;
}
QSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 3px; /* Match main radius */

}
QSpinBox::down-button {
    subcontrol-position: bottom right;
     border-bottom-right-radius: 3px; /* Match main radius */
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
     background-color: #666666;
}

/* Basic arrows - consider SVG/Font icons for better look */
QSpinBox::up-arrow {
    image: url(./icons/arrow_up.png); /* Placeholder - create or find an icon */
    width: 10px; height: 10px;
}
QSpinBox::down-arrow {
    image: url(./icons/arrow_down.png); /* Placeholder */
     width: 10px; height: 10px;
}


QCheckBox {
    spacing: 5px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    background-color: #3c3f41;
    border: 2px solid #555555;
    border-radius: 4px;
}

QCheckBox::indicator:checked {
    background-color: #55aaff;
    border: 2px solid #66bbff;
    image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDE2IDE2Ij48cGF0aCBmaWxsPSIjZmZmZmZmIiBkPSJNNi41IDEyLjVsLTQtNC0xLjUgMS41IDYgNiA4LTggMS41LTEuNXoiLz48L3N2Zz4=);
}

QCheckBox::indicator:unchecked:hover {
    border: 2px solid #666666;
    background-color: #444444;
}

QCheckBox::indicator:checked:hover {
    background-color: #66bbff;
    border: 2px solid #77ccff;
}

QCheckBox:focus {
    outline: none;
}

QCheckBox:focus::indicator {
    border: 2px solid #77ccff;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    text-align: center;
    color: #f0f0f0;
    background-color: #3c3f41;
}

QProgressBar::chunk {
    background-color: #55aaff; /* Accent color */
    border-radius: 3px; /* Slightly smaller radius for the chunk */
    margin: 0.5px; /* Small margin */
}

QLabel {
    background-color: transparent; /* Inherit from parent */
    color: #f0f0f0;
}

QTextEdit {
    /* Already styled above */
}

QMessageBox {
     background-color: #3c3f41;
}

QMessageBox QLabel {
    color: #f0f0f0;
}

QMessageBox QPushButton {
     min-width: 60px; /* Smaller buttons in message box */
     padding: 4px 8px;
}

QFileDialog {
    /* Usually uses native look, but some elements might be stylable */
    background-color: #3c3f41; /* Might not fully work */
}

QTabWidget::pane {
    border: 1px solid #555555;
    background-color: #3c3f41; /* Match background */
    margin-top: -1px; /* Overlap border with tab bar */
}

QTabBar::tab {
    background-color: #444444;
    color: #f0f0f0;
    border: 1px solid #555555;
    border-bottom: none; /* Remove bottom border */
    padding: 6px 12px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #3c3f41; /* Match pane background */
    border-color: #555555;
    margin-bottom: -1px; /* Overlap the pane border */
}

QTabBar::tab:!selected:hover {
    background-color: #555555;
}

"""

# --- Helper Functions ---
def is_admin():
    """Check if the script is running with Administrator privileges on Windows."""
    try:
        # Ensure this runs only on Windows
        if os.name == 'nt':
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            # On non-Windows, assume not admin for VHDX purposes
            # Real check for root on Unix would use os.geteuid() == 0
            return False
    except Exception:
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
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        # Ignore files we can't access/getsize for
                        pass
    except OSError:
        return -1 # Indicate error
    return total_size

def find_iso_tool():
    """Tries to find mkisofs, genisoimage, or xorriso"""
    tools_to_try = ['xorriso', 'mkisofs', 'genisoimage']
    for tool in tools_to_try:
        path = shutil.which(tool)
        if path:
            return path, tool # Return path and name
    return None, None

# --- Worker Thread Base Class ---
class BaseWorker(QObject):
    """Base class for worker threads with common signals."""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # bool: success, str: message
    is_cancelled = False

    def __init__(self):
        super().__init__()

    def _decode_output(self, output_bytes):
        """Decodes bytes using utf-8, replacing errors."""
        if not output_bytes:
            return ""
        try:
            # Try UTF-8 first, most common for modern tools
            return output_bytes.decode('utf-8', errors='replace')
        except Exception:
            try:
                # Fallback to system's default encoding (often MBCS on Windows)
                 return output_bytes.decode(sys.getdefaultencoding(), errors='replace')
            except Exception:
                # Final fallback, just represent bytes directly
                return repr(output_bytes)

# --- Worker Thread for VHDX Operations ---
class VHDXWorker(BaseWorker):
    """
    Handles VHDX creation, formatting, copying, and mounting
    in a separate thread.
    """
    def __init__(self, source_folder, output_vhdx, vhdx_size_gb, mount_vhdx):
        super().__init__()
        self.source_folder = Path(source_folder)
        self.output_vhdx = Path(output_vhdx)
        self.vhdx_size_bytes = vhdx_size_gb * 1024**3
        self.mount_vhdx = mount_vhdx
        self.drive_letter = None

    def _run_powershell(self, command, check_errors=True):
        """Executes a PowerShell command and returns decoded stdout."""
        if self.is_cancelled: raise RuntimeError("Operation Cancelled")
        if os.name != 'nt': raise RuntimeError("VHDX operations require Windows.")

        full_command = f'powershell.exe -ExecutionPolicy Bypass -NonInteractive -NoProfile -Command "{command}"'
        # self.status_update.emit(f"DEBUG PS: {full_command[:200]}") # Debug: Log command

        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                shell=True, # Often needed for complex PS commands/pipelines
                check=False # Manual check below
            )

            stdout_str = self._decode_output(result.stdout)
            stderr_str = self._decode_output(result.stderr)

            # self.status_update.emit(f"DEBUG PS OUT: {stdout_str}") # Debug
            # self.status_update.emit(f"DEBUG PS ERR: {stderr_str}") # Debug

            if check_errors and result.returncode != 0:
                error_msg = f"PowerShell Error (Code {result.returncode}):\n{stderr_str or stdout_str or 'No output'}"
                raise RuntimeError(error_msg)

            return stdout_str.strip()

        except FileNotFoundError:
            raise RuntimeError("Error: powershell.exe not found. Is PowerShell installed and in PATH?")
        except Exception as e:
            # Catch potential decode errors during the decode() step itself if bytes were truly malformed
            raise RuntimeError(f"PowerShell execution or output processing failed: {e}")


    def _parse_drive_letter(self, format_output):
        """ Tries to parse the drive letter from Format-Volume output or query it."""
        # Try direct output parsing first
        match = re.search(r"DriveLetter\s+:\s+([A-Z])", format_output, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: Query via Get-Partition -> Get-Volume
        self.status_update.emit("Attempting to query drive letter...")
        try:
            # Ensure VHD is still mounted before querying
            vhdx_path_escaped = str(self.output_vhdx).replace('"', '`"') # Basic PowerShell escape
            check_mount_cmd = f'(Get-VHD -Path \\"{vhdx_path_escaped}\\").Attached'
            is_attached = self._run_powershell(check_mount_cmd, check_errors=False).strip().lower() == 'true'
            if not is_attached:
                self.status_update.emit("Warning: VHDX appears detached, cannot query drive letter.")
                return None

            disk_num_command = f'(Get-VHD -Path \\"{vhdx_path_escaped}\\").DiskNumber'
            disk_num_output = self._run_powershell(disk_num_command)
            if disk_num_output and disk_num_output.isdigit():
                disk_num = int(disk_num_output)
                # Get the first partition, then its volume's drive letter
                volume_command = f'Get-Partition -DiskNumber {disk_num} | Select-Object -First 1 | Get-Volume | Select-Object -ExpandProperty DriveLetter'
                drive_letter_output = self._run_powershell(volume_command, check_errors=False) # Don't fail if no letter assigned yet
                if drive_letter_output and len(drive_letter_output) == 1 and drive_letter_output.isalpha():
                    cleaned_letter = ''.join(filter(str.isalpha, drive_letter_output))
                    if len(cleaned_letter) == 1:
                       return cleaned_letter.upper()
        except Exception as e:
            self.status_update.emit(f"Warning: Could not reliably determine drive letter after format: {e}")
        return None


    def _copy_files_with_progress(self, src, dest):
        """Uses Robocopy for reliable copying."""
        self.status_update.emit("Starting file copy process (using Robocopy)...")
        src_str = str(src)
        dest_str = str(dest).rstrip('\\') # Robocopy works best without trailing slash

        # Ensure destination drive is ready - small delay can help
        time.sleep(3)
        if not Path(dest_str).exists():
             self.status_update.emit(f"Warning: Mount point {dest_str} not found before Robocopy. Retrying mount check...")
             # Add a re-check or fail? For now, proceed and let Robocopy fail if needed.
             # Re-parse might be needed if the letter changed or wasn't found initially
             if not self.drive_letter:
                 raise RuntimeError("Cannot copy files: Drive letter is unknown.")
             dest_str = f"{self.drive_letter}:\\"


        log_file = self.output_vhdx.with_suffix(".robocopy.log")
        # Escape paths for the command line (simple quoting)
        command = [
            'robocopy',
            f'"{src_str}"',
            f'"{dest_str}"',
            '/E',       # Copy subdirectories, including empty ones
            '/COPYALL', # Copy all file info (Data, Attributes, Timestamps, Security, Owner, Auditing info)
            '/ETA',     # Show Estimated Time Remaining
            '/R:2',     # Retry twice on errors
            '/W:5',     # Wait 5 seconds between retries
            '/NP',      # No Progress - Don't show % progress per file (cleaner log)
            '/NFL',     # No File List - Don't log file names
            '/NDL',     # No Directory List - Don't log directory names
            '/TEE',     # Output to console window AND log file
            f'/LOG+:"{log_file}"' # Append to log file
        ]
        command_str = " ".join(command)
        self.status_update.emit(f"Robocopy command (log: {log_file.name})")

        try:
            process = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout_bytes, stderr_bytes = process.communicate() # Wait for completion

            # Decode output leniently
            stdout = self._decode_output(stdout_bytes)
            stderr = self._decode_output(stderr_bytes)

            # Log Robocopy summary (often at the end of stdout)
            self.status_update.emit("--- Robocopy Summary ---")
            # Find the summary table lines
            summary_lines = [line for line in stdout.splitlines() if "Total" in line or "Copied" in line or "Skipped" in line or "Mismatch" in line or "FAILED" in line or "Extras" in line]
            if summary_lines:
                 for line in summary_lines[-10:]: # Show last few lines likely containing the summary
                      self.status_update.emit(line.strip())
            else:
                 # Show tail if no obvious summary found
                 self.status_update.emit(f"Output (last 500 chars):\n{stdout[-500:].strip()}")
            if stderr: self.status_update.emit(f"Stderr:\n{stderr.strip()}")
            self.status_update.emit("--- End Robocopy Summary ---")


            # Robocopy return codes:
            # 0 = No errors, no files copied
            # 1 = No errors, files copied
            # 2 = Extra files/dirs detected
            # 3 = 1 + 2
            # 4 = Mismatched files/dirs detected
            # 5 = 1 + 4
            # 6 = 2 + 4
            # 7 = 1 + 2 + 4
            # 8+ = Failure
            if process.returncode >= 8:
                 raise RuntimeError(f"Robocopy failed with critical error code {process.returncode}. Check log: {log_file}")
            elif process.returncode > 1:
                 self.status_update.emit(f"Robocopy completed with code {process.returncode} (Info: Some files might be skipped/extra/mismatched). Check log: {log_file}")
            else:
                 self.status_update.emit("Robocopy file copy completed successfully.")

        except FileNotFoundError:
            raise RuntimeError("Error: robocopy.exe not found. Is it installed and in PATH? (Standard on modern Windows)")
        except Exception as e:
            raise RuntimeError(f"File copy execution failed: {e}")


    def run(self):
        """Main execution logic for the VHDX worker thread."""
        try:
            if not is_admin():
                raise RuntimeError("Administrator privileges are required to create and mount VHDX files.")
            if os.name != 'nt':
                 raise RuntimeError("VHDX operations are only supported on Windows.")

            self.status_update.emit(f"Source: {self.source_folder}")
            self.status_update.emit(f"Output: {self.output_vhdx}")
            self.status_update.emit(f"Size: {self.vhdx_size_bytes / 1024**3:.2f} GB")
            self.status_update.emit(f"Mount after creation: {self.mount_vhdx}")
            self.progress_update.emit(0)

            # --- 1. Check Output Directory & Overwrite ---
            if not self.output_vhdx.parent.exists():
                self.status_update.emit(f"Creating output directory: {self.output_vhdx.parent}")
                self.output_vhdx.parent.mkdir(parents=True, exist_ok=True)
            if self.output_vhdx.exists():
                self.status_update.emit(f"Output file '{self.output_vhdx}' exists, deleting...")
                try:
                    self.output_vhdx.unlink()
                except Exception as e:
                     # Maybe it's mounted? Try dismount first.
                     self.status_update.emit(f"Could not delete existing file, attempting dismount first: {e}")
                     try:
                          dismount_cmd = f'Dismount-VHD -Path \\"{str(self.output_vhdx).replace('"', '`"')}\\" -ErrorAction SilentlyContinue'
                          self._run_powershell(dismount_cmd, check_errors=False)
                          self.output_vhdx.unlink() # Try deleting again
                          self.status_update.emit("Existing file deleted after dismount.")
                     except Exception as e2:
                          raise RuntimeError(f"Failed to delete existing VHDX file '{self.output_vhdx}'. Is it in use? Error: {e2}")


            # --- 2. Create VHDX ---
            self.status_update.emit("Step 1/3: Creating VHDX file...")
            vhdx_create_command = f'New-VHD -Path "{self.output_vhdx}" -SizeBytes {self.vhdx_size_bytes} -BlockSizeBytes 1MB -Confirm:$false'
            self._run_powershell(vhdx_create_command)
            self.status_update.emit("✓ VHDX file created successfully.")
            self.progress_update.emit(15)

            # --- 3. Mount VHDX ---
            self.status_update.emit("Step 2/3: Mounting VHDX...")
            vhdx_path_escaped = str(self.output_vhdx).replace('"', '`"')
            mount_command = f'$mountResult = Mount-VHD -Path \\"{vhdx_path_escaped}\\" -Passthru -ErrorAction Stop; $mountResult.DiskNumber'
            disk_number_output = self._run_powershell(mount_command)
            if not disk_number_output or not disk_number_output.isdigit():
                 raise RuntimeError(f"Failed to get disk number after mounting. Output: {disk_number_output}")
            disk_number = int(disk_number_output)
            self.status_update.emit(f"✓ VHDX mounted as Disk {disk_number}.")
            self.progress_update.emit(30)

            # --- 4. Initialize & Partition Disk ---
            self.status_update.emit(f"Step 2/3: Formatting disk (this may take a moment)...")
            init_command = f'Initialize-Disk -Number {disk_number} -PartitionStyle GPT -Confirm:$false -ErrorAction Stop'
            try:
                 self._run_powershell(init_command)
                 self.status_update.emit("✓ Disk initialized with GPT partition style.")
            except RuntimeError as e:
                if "already been initialized" in str(e).lower():
                     self.status_update.emit("ℹ Disk already initialized (continuing).")
                else:
                    raise e

            self.status_update.emit("Creating and formatting partition...")
            partition_command = (
                f'$partition = New-Partition -DiskNumber {disk_number} -UseMaximumSize -AssignDriveLetter -ErrorAction Stop; '
                f'$partition | Format-Volume -FileSystem NTFS -Confirm:$false -Force -ErrorAction Stop'
            )
            format_output = self._run_powershell(partition_command)
            self.progress_update.emit(50)

            # Get drive letter
            time.sleep(3)
            self.drive_letter = self._parse_drive_letter(format_output)
            if self.drive_letter:
                 self.status_update.emit(f"✓ Volume formatted successfully as Drive {self.drive_letter}:")
            else:
                 time.sleep(5)
                 self.drive_letter = self._parse_drive_letter("")
                 if self.drive_letter:
                      self.status_update.emit(f"✓ Volume formatted successfully as Drive {self.drive_letter}:")
                 else:
                      raise RuntimeError("Failed to create/format partition or determine drive letter after multiple attempts.")

            self.progress_update.emit(55)

            # --- 5. Copy Files ---
            self.status_update.emit("Step 3/3: Copying files...")
            mount_point = Path(f"{self.drive_letter}:\\")
            self.status_update.emit(f"Copying files from '{self.source_folder}' to '{mount_point}'...")
            self._copy_files_with_progress(self.source_folder, mount_point)
            self.progress_update.emit(95)

            # --- 6. Dismount (Optional) ---
            final_message = f"VHDX created successfully at '{self.output_vhdx}'"
            if self.mount_vhdx:
                final_message += f" and mounted as Drive {self.drive_letter}:"
                self.status_update.emit("Keeping VHDX mounted as requested.")
            else:
                self.status_update.emit("Dismounting VHDX...")
                dismount_command = f'Dismount-VHD -Path \\"{vhdx_path_escaped}\\" -ErrorAction Stop'
                self._run_powershell(dismount_command)
                self.status_update.emit("VHDX dismounted.")
                final_message += ". (Not mounted)"

            self.progress_update.emit(100)
            self.status_update.emit("Operation completed successfully.")
            self.finished.emit(True, final_message)

        except Exception as e:
            error_message = f"ERROR: {e}"
            self.status_update.emit(error_message)
             # Attempt cleanup: Dismount if it was mounted and not requested to stay mounted
            if self.drive_letter and not self.mount_vhdx:
                 try:
                     self.status_update.emit("Attempting to dismount VHDX after error...")
                     vhdx_path_escaped = str(self.output_vhdx).replace('"', '`"')
                     dismount_command = f'Dismount-VHD -Path \\"{vhdx_path_escaped}\\" -ErrorAction SilentlyContinue'
                     self._run_powershell(dismount_command, check_errors=False)
                     self.status_update.emit("Dismount attempt finished.")
                 except Exception as cleanup_e:
                     self.status_update.emit(f"Could not dismount VHDX during cleanup: {cleanup_e}")

            self.progress_update.emit(0) # Reset progress on failure
            self.finished.emit(False, f"Operation failed: {e}")

# --- Worker Thread for ISO Operations ---
class ISOWorker(BaseWorker):
    """
    Handles the ISO creation in a separate thread.
    Requires mkisofs, genisoimage, or xorriso in PATH.
    """
    def __init__(self, source_folder, output_iso, volume_label):
        super().__init__()
        self.source_folder = Path(source_folder)
        self.output_iso = Path(output_iso)
        # Sanitize volume label - limit length and chars (ISO 9660 L1 basic)
        sanitized_label = re.sub(r'[^A-Za-z0-9_]', '_', volume_label)[:32] if volume_label else ""
        self.volume_label = sanitized_label
        self.iso_tool_path = None
        self.iso_tool_name = None

    def _run_iso_command(self, command_list):
        """Executes the ISO creation command."""
        if self.is_cancelled: raise RuntimeError("Operation Cancelled")

        self.status_update.emit(f"Executing: {' '.join(command_list)}")
        try:
            process = subprocess.Popen(
                command_list, # Pass as list
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False, # Safer not to use shell when command is a list
            )
            stdout_bytes, stderr_bytes = process.communicate()

            stdout_str = self._decode_output(stdout_bytes)
            stderr_str = self._decode_output(stderr_bytes)

            # Log output
            self.status_update.emit("--- ISO Tool Output ---")
            if stdout_str: self.status_update.emit(f"Output:\n{stdout_str.strip()}")
            if stderr_str: self.status_update.emit(f"Warnings/Errors:\n{stderr_str.strip()}")
            self.status_update.emit("--- End ISO Tool Output ---")

            # Check return code AFTER logging output
            if process.returncode != 0:
                error_msg = f"ISO Tool '{self.iso_tool_name}' failed (Code {process.returncode}). Check log above."
                raise RuntimeError(error_msg)

            return stdout_str.strip()

        except FileNotFoundError:
            raise RuntimeError(f"Error: ISO tool '{command_list[0]}' not found or failed to execute.")
        except Exception as e:
            raise RuntimeError(f"ISO tool execution failed: {e}")

    def run(self):
        """Main execution logic for the ISO worker thread."""
        try:
            self.iso_tool_path, self.iso_tool_name = find_iso_tool()
            if not self.iso_tool_path:
                raise RuntimeError("Could not find 'mkisofs', 'genisoimage', or 'xorriso' in system PATH. Please install one (e.g., cdrtools or libisoburn).")
            self.status_update.emit(f"Using ISO tool: {self.iso_tool_path}")

            self.status_update.emit(f"Source: {self.source_folder}")
            self.status_update.emit(f"Output: {self.output_iso}")
            self.status_update.emit(f"Volume Label: {self.volume_label or '(Default)'}")
            self.progress_update.emit(0)

            # --- Check Output Directory & Overwrite ---
            if not self.output_iso.parent.exists():
                self.status_update.emit(f"Creating output directory: {self.output_iso.parent}")
                self.output_iso.parent.mkdir(parents=True, exist_ok=True)
            if self.output_iso.exists():
                self.status_update.emit(f"Output file '{self.output_iso}' exists, deleting...")
                try:
                    self.output_iso.unlink()
                except Exception as e:
                    raise RuntimeError(f"Failed to delete existing ISO file '{self.output_iso}'. Error: {e}")


            # --- Construct command ---
            # Use common mkisofs args, often compatible with xorriso via emulation
            command = [
                self.iso_tool_path,
                "-o", str(self.output_iso), # Output file
                "-R",               # Rock Ridge extensions (Unix permissions, long names)
                "-J",               # Joliet extensions (Windows long names)
                "-joliet-long",     # Allow longer Joliet names
                "-appid", APP_NAME, # Application ID
                "-publisher", f"Created by {APP_NAME}", # Publisher info
            ]
            if self.volume_label:
                command.extend(["-V", self.volume_label]) # Volume ID

            # Source directory MUST be the last argument
            command.append(str(self.source_folder))

            # Set indeterminate progress or stages
            self.progress_update.emit(10)
            self.status_update.emit("Creating ISO file... This may take some time depending on size.")

            # --- Execute ISO command ---
            self._run_iso_command(command)

            # Assume success if command didn't raise error
            self.progress_update.emit(100)
            self.status_update.emit("ISO creation completed successfully.")
            self.finished.emit(True, f"ISO file created successfully at '{self.output_iso}'")

        except Exception as e:
            error_message = f"ERROR: {e}"
            self.status_update.emit(error_message)
            self.progress_update.emit(0) # Reset progress on error
            self.finished.emit(False, f"ISO creation failed: {e}")


# --- Main Application Window ---
class VHDXCreatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_worker_thread = None
        self.active_worker = None
        self.iso_tool_path = None
        self.iso_tool_name = None
        self.init_ui()
        self.apply_styles()


    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.setMinimumSize(650, 550)
        # self.setWindowIcon(QIcon("path/to/your/icon.ico")) # Optional

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Create VHDX Tab ---
        self.vhdx_tab = QWidget()
        self.tab_widget.addTab(self.vhdx_tab, "VHDX Creator")
        vhdx_layout = QVBoxLayout(self.vhdx_tab)
        self.setup_vhdx_ui(vhdx_layout)

        # --- Create ISO Tab ---
        self.iso_tab = QWidget()
        self.tab_widget.addTab(self.iso_tab, "ISO Creator")
        iso_layout = QVBoxLayout(self.iso_tab)
        self.setup_iso_ui(iso_layout)

        # --- Shared Controls ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status Log:")
        self.status_output = QTextEdit()
        self.status_output.setReadOnly(True)
        self.status_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth) # Wrap lines
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.status_output, 1) # Stretch factor

        self.create_button = QPushButton("Create")
        self.create_button.clicked.connect(self.start_creation)
        self.create_button.setObjectName("CreateButton") # For potential specific styling
        self.create_button.setMinimumHeight(30) # Make button slightly taller
        main_layout.addWidget(self.create_button, 0, Qt.AlignmentFlag.AlignRight)

        # Initial status messages
        self.update_status("Ready. Select operation type and provide details.")
        if os.name == 'nt' and not is_admin():
             self.update_status("WARNING: App not running as Administrator. VHDX operations will fail.")
        self.check_iso_tool_presence(silent=True) # Check silently on startup


    def apply_styles(self):
        """Applies the QSS stylesheet."""
        try:
            # Try loading from external file first (allows user customization)
            qss_file = Path(__file__).parent / "dark_theme.qss"
            if qss_file.exists():
                 with open(qss_file, "r") as f:
                      self.setStyleSheet(f.read())
                      print("Applied stylesheet from dark_theme.qss")
            else:
                 # Fallback to internal QSS string
                 self.setStyleSheet(DARK_THEME_QSS)
                 print("Applied internal dark theme stylesheet.")
        except Exception as e:
            print(f"Error applying stylesheet: {e}. Using default style.")
            # Optional: Apply a basic programmatic dark palette as a fallback
            # app = QApplication.instance()
            # if app:
            #     app.setStyle("Fusion")
            #     palette = QPalette()
            #     palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            #     # ... set other colors ...
            #     app.setPalette(palette)


    def setup_vhdx_ui(self, layout):
        """Creates UI elements for the VHDX tab."""
        source_group = QGroupBox("Source Folder")
        source_layout = QHBoxLayout(source_group)
        self.vhdx_source_path_edit = QLineEdit()
        self.vhdx_source_path_edit.setPlaceholderText("Select folder to include in VHDX...")
        self.vhdx_source_path_edit.setReadOnly(True)
        self.vhdx_source_browse_button = QPushButton("Browse...")
        self.vhdx_source_browse_button.clicked.connect(lambda: self.select_folder(self.vhdx_source_path_edit))
        source_layout.addWidget(self.vhdx_source_path_edit)
        source_layout.addWidget(self.vhdx_source_browse_button)
        layout.addWidget(source_group)

        output_group = QGroupBox("Output VHDX File")
        output_layout = QHBoxLayout(output_group)
        self.vhdx_output_path_edit = QLineEdit()
        self.vhdx_output_path_edit.setPlaceholderText("Select output path for .vhdx file...")
        self.vhdx_output_browse_button = QPushButton("Browse...")
        self.vhdx_output_browse_button.clicked.connect(self.select_output_vhdx)
        output_layout.addWidget(self.vhdx_output_path_edit)
        output_layout.addWidget(self.vhdx_output_browse_button)
        layout.addWidget(output_group)

        options_group = QGroupBox("VHDX Options")
        options_layout = QHBoxLayout(options_group)
        self.size_label = QLabel("Size (GB):")
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(MIN_VHDX_SIZE_GB, MAX_VHDX_SIZE_GB)
        self.size_spinbox.setValue(DEFAULT_VHDX_SIZE_GB)
        self.size_spinbox.setToolTip(f"Size of the virtual disk ({MIN_VHDX_SIZE_GB}-{MAX_VHDX_SIZE_GB} GB)")
        self.mount_checkbox = QCheckBox("Mount after creation")
        self.mount_checkbox.setChecked(True)
        self.mount_checkbox.setToolTip("Automatically attach the VHDX as a drive after creation")
        self.mount_checkbox.stateChanged.connect(self.on_mount_checkbox_changed)
        options_layout.addWidget(self.size_label)
        options_layout.addWidget(self.size_spinbox)
        options_layout.addStretch(1)
        options_layout.addWidget(self.mount_checkbox)
        layout.addWidget(options_group)

        layout.addStretch(1) # Push elements up


    def setup_iso_ui(self, layout):
        """Creates UI elements for the ISO tab."""
        source_group = QGroupBox("Source Folder")
        source_layout = QHBoxLayout(source_group)
        self.iso_source_path_edit = QLineEdit()
        self.iso_source_path_edit.setPlaceholderText("Select folder to build ISO from...")
        self.iso_source_path_edit.setReadOnly(True)
        self.iso_source_browse_button = QPushButton("Browse...")
        self.iso_source_browse_button.clicked.connect(lambda: self.select_folder(self.iso_source_path_edit))
        source_layout.addWidget(self.iso_source_path_edit)
        source_layout.addWidget(self.iso_source_browse_button)
        layout.addWidget(source_group)

        output_group = QGroupBox("Output ISO File")
        output_layout = QHBoxLayout(output_group)
        self.iso_output_path_edit = QLineEdit()
        self.iso_output_path_edit.setPlaceholderText("Select output path for .iso file...")
        self.iso_output_browse_button = QPushButton("Browse...")
        self.iso_output_browse_button.clicked.connect(self.select_output_iso)
        output_layout.addWidget(self.iso_output_path_edit)
        output_layout.addWidget(self.iso_output_browse_button)
        layout.addWidget(output_group)

        options_group = QGroupBox("ISO Options")
        options_layout = QHBoxLayout(options_group)
        self.volume_label_label = QLabel("Volume Label:")
        self.volume_label_edit = QLineEdit()
        self.volume_label_edit.setPlaceholderText("ISO volume name (optional)")
        self.volume_label_edit.setMaxLength(32)
        self.volume_label_edit.setToolTip("Disc name when mounted (max 32 chars, A-Z, 0-9, _ recommended)")
        options_layout.addWidget(self.volume_label_label)
        options_layout.addWidget(self.volume_label_edit)
        layout.addWidget(options_group)

        self.iso_tool_status_label = QLabel("ISO Tool Status: Checking...")
        self.iso_tool_status_label.setStyleSheet("font-style: italic; color: #aaa;")
        layout.addWidget(self.iso_tool_status_label)
        layout.addStretch(1)


    # --- File/Folder Selection ---
    def select_folder(self, target_line_edit):
        """Opens a directory dialog."""
        current_path = target_line_edit.text()
        start_dir = str(Path(current_path).parent if current_path else Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder", start_dir)
        if folder:
            target_line_edit.setText(folder)

    def select_output_vhdx(self):
        default_name = "MyDisk.vhdx"
        current_path = self.vhdx_output_path_edit.text()
        start_dir = str(Path(current_path).parent if current_path else Path.home())
        default_path = str(Path(start_dir) / default_name)

        filepath, _ = QFileDialog.getSaveFileName(self, "Save VHDX File", default_path, "VHDX Files (*.vhdx);;All Files (*)")
        if filepath:
            if not filepath.lower().endswith(".vhdx"): filepath += ".vhdx"
            self.vhdx_output_path_edit.setText(filepath)

    def select_output_iso(self):
        default_name = "MyImage.iso"
        current_path = self.iso_output_path_edit.text()
        start_dir = str(Path(current_path).parent if current_path else Path.home())
        default_path = str(Path(start_dir) / default_name)

        filepath, _ = QFileDialog.getSaveFileName(self, "Save ISO File", default_path, "ISO Files (*.iso);;All Files (*)")
        if filepath:
            if not filepath.lower().endswith(".iso"): filepath += ".iso"
            self.iso_output_path_edit.setText(filepath)


    # --- Status and Progress Updates ---
    def update_status(self, message):
        """Appends a timestamped message to the status log."""
        timestamp = time.strftime('%H:%M:%S')
        self.status_output.append(f"[{timestamp}] {message}")
        # Auto-scroll to the bottom
        scrollbar = self.status_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def set_controls_enabled(self, enabled):
        """Enable/disable controls based on operation state."""
        # Disable tab switching while running
        self.tab_widget.setEnabled(enabled)
        # Disable all specific controls within tabs
        for widget in self.vhdx_tab.findChildren(QWidget):
            widget.setEnabled(enabled)
        for widget in self.iso_tab.findChildren(QWidget):
            widget.setEnabled(enabled)
        # Control the main button
        self.create_button.setEnabled(enabled)


    # --- Worker Management ---
    def creation_finished(self, success, message):
        """Handles completion signals from workers."""
        self.update_status(f"--- Operation Finished ---")
        self.update_status(message)
        self.set_controls_enabled(True)

        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Success", message)
        else:
            # Don't reset progress bar immediately on failure, keep last value
            # self.progress_bar.setValue(0)
            self.status_output.verticalScrollBar().setValue(self.status_output.verticalScrollBar().maximum())
            QMessageBox.critical(self, "Error", f"Operation failed.\n\n{message}\n\nCheck the status log for details.")

        # Clean up worker and thread
        if self.active_worker_thread:
            self.active_worker_thread.quit()
            if not self.active_worker_thread.wait(3000): # Wait max 3 secs
                 self.update_status("Warning: Worker thread did not terminate cleanly. Forcing termination.")
                 self.active_worker_thread.terminate() # Force if necessary
                 self.active_worker_thread.wait() # Wait after terminate

            self.active_worker = None
            self.active_worker_thread = None
        self.update_status("Ready for next operation.")


    def check_iso_tool_presence(self, silent=False):
        """Checks for ISO tool and updates status label."""
        self.iso_tool_path, self.iso_tool_name = find_iso_tool()
        if self.iso_tool_path:
            status_text = f"ISO Tool Status: Found '{self.iso_tool_name}' at {self.iso_tool_path}"
            if not silent: self.update_status(status_text)
            self.iso_tool_status_label.setText(f"✓ Found: {self.iso_tool_name}")
            self.iso_tool_status_label.setStyleSheet("font-style: normal; color: #77cc77;") # Greenish
            return True
        else:
            status_text = "ISO Tool Status: Not Found (Install mkisofs, genisoimage, or xorriso)"
            if not silent:
                self.update_status(f"WARNING: {status_text}")
                QMessageBox.warning(self, "ISO Tool Missing",
                                    "Could not find 'xorriso', 'mkisofs', or 'genisoimage' in your system's PATH.\n"
                                    "ISO creation is disabled until one is installed.")
            self.iso_tool_status_label.setText(f"✗ Not Found")
            self.iso_tool_status_label.setStyleSheet("font-style: italic; color: #cc7777;") # Reddish
            return False


    # --- Main Action ---
    def start_creation(self):
        """Validates inputs and starts the appropriate worker thread."""
        current_index = self.tab_widget.currentIndex()
        self.status_output.clear()
        self.progress_bar.setValue(0)

        if current_index == 0: # VHDX Tab
            self.start_vhdx_creation()
        elif current_index == 1: # ISO Tab
            self.start_iso_creation()
        else:
            QMessageBox.critical(self, "Internal Error", "Invalid tab selected.")


    def validate_paths(self, source_path_str, output_path_str, type_name):
        """Common path validation logic."""
        source_path = Path(source_path_str)
        output_path = Path(output_path_str)

        if not source_path_str or not source_path.is_dir():
            QMessageBox.warning(self, "Input Error", f"{type_name}: Please select a valid source folder.")
            return None, None, False

        if not output_path_str:
            QMessageBox.warning(self, "Input Error", f"{type_name}: Please specify a valid output file path.")
            return None, None, False

        if not output_path.parent.exists():
            try:
                self.update_status(f"Creating non-existent output directory: {output_path.parent}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Input Error", f"{type_name}: Output directory cannot be created:\n{output_path.parent}\nError: {e}")
                return None, None, False

        # Check overwrite AFTER other checks
        if output_path.exists():
            reply = QMessageBox.question(self, "File Exists",
                                         f"The file '{output_path.name}' already exists in:\n{output_path.parent}\n\nDo you want to overwrite it?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return None, None, False
            # Will attempt delete in worker if Yes

        return source_path, output_path, True


    def start_vhdx_creation(self):
        self.update_status("Validating VHDX inputs...")
        source_folder_str = self.vhdx_source_path_edit.text()
        output_vhdx_str = self.vhdx_output_path_edit.text()
        vhdx_size_gb = self.size_spinbox.value()
        mount_vhdx = self.mount_checkbox.isChecked()

        source_path, output_path, paths_valid = self.validate_paths(source_folder_str, output_vhdx_str, "VHDX")
        if not paths_valid: return

        # VHDX specific checks
        if os.name != 'nt':
             QMessageBox.critical(self, "Platform Error", "VHDX creation is only supported on Windows.")
             return
        if not is_admin():
            QMessageBox.critical(self, "Permissions Error", "VHDX creation requires Administrator privileges.\nPlease restart the application as an Administrator.")
            return
        if vhdx_size_gb < MIN_VHDX_SIZE_GB:
             QMessageBox.warning(self, "Input Error", f"VHDX size must be at least {MIN_VHDX_SIZE_GB} GB.")
             return

        # Disk space checks (optional but good)
        if PSUTIL_AVAILABLE:
            try:
                free_space = psutil.disk_usage(str(output_path.parent)).free
                required_space = vhdx_size_gb * (1024**3)
                if free_space < required_space * 1.05: # Need space for VHDX file + ~5% margin
                     QMessageBox.warning(self, "Disk Space Low",
                                        f"Insufficient disk space on drive {output_path.drive or output_path.parent}.\n"
                                        f"Required for VHDX file: ~{required_space / 1024**3:.2f} GB\n"
                                        f"Available: {free_space / 1024**3:.2f} GB")
                     return # Stop if likely insufficient space
            except Exception as e:
                 self.update_status(f"Warning: Could not check disk space: {e}")

        # Source size vs VHDX capacity check
        try:
             src_size = get_folder_size(source_path)
             if src_size < 0:
                  self.update_status("Warning: Could not reliably determine source folder size.")
             # Compare bytes (allow ~5% FS overhead, maybe more for smaller disks)
             elif src_size > vhdx_size_gb * (1024**3) * 0.95:
                 reply = QMessageBox.question(self, "Size Warning",
                                         f"The source folder size (~{src_size / 1024**3:.2f} GB) "
                                         f"might exceed the capacity of the VHDX ({vhdx_size_gb} GB) after formatting.\n\n"
                                         f"This could cause the file copy to fail.\nContinue anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.No:
                     return
        except Exception as e:
            self.update_status(f"Warning: Could not check source folder size: {e}")

        # --- Start VHDX Worker ---
        self.update_status("Inputs validated. Starting VHDX creation...")
        self.set_controls_enabled(False)
        self.active_worker_thread = QThread()
        self.active_worker = VHDXWorker(str(source_path), str(output_path), vhdx_size_gb, mount_vhdx)
        self.active_worker.moveToThread(self.active_worker_thread)

        # Connect signals
        self.active_worker.progress_update.connect(self.update_progress)
        self.active_worker.status_update.connect(self.update_status)
        self.active_worker.finished.connect(self.creation_finished)
        # Ensure thread quits when worker is done
        self.active_worker.finished.connect(self.active_worker_thread.quit)
        self.active_worker_thread.started.connect(self.active_worker.run)
        self.active_worker_thread.start()


    def start_iso_creation(self):
        self.update_status("Validating ISO inputs...")
        source_folder_str = self.iso_source_path_edit.text()
        output_iso_str = self.iso_output_path_edit.text()
        volume_label = self.volume_label_edit.text().strip()

        source_path, output_path, paths_valid = self.validate_paths(source_folder_str, output_iso_str, "ISO")
        if not paths_valid: return

        # ISO specific checks
        if not self.check_iso_tool_presence(silent=False): # Non-silent check before starting
            return

        # Optional: More strict volume label validation?
        # Current worker sanitizes, this is just UI feedback if needed
        # allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        # if volume_label and not set(volume_label).issubset(allowed_chars):
        #      # Warn user? Worker will sanitize anyway.
        #      pass

        # --- Start ISO Worker ---
        self.update_status("Inputs validated. Starting ISO creation...")
        self.set_controls_enabled(False)
        self.active_worker_thread = QThread()
        self.active_worker = ISOWorker(str(source_path), str(output_path), volume_label)
        self.active_worker.moveToThread(self.active_worker_thread)

        # Connect signals
        self.active_worker.progress_update.connect(self.update_progress)
        self.active_worker.status_update.connect(self.update_status)
        self.active_worker.finished.connect(self.creation_finished)
        # Ensure thread quits when worker is done
        self.active_worker.finished.connect(self.active_worker_thread.quit)
        self.active_worker_thread.started.connect(self.active_worker.run)
        self.active_worker_thread.start()


    def closeEvent(self, event):
        """Handle closing while an operation is running."""
        if self.active_worker_thread and self.active_worker_thread.isRunning():
            reply = QMessageBox.question(self, "Operation in Progress",
                                         "An operation is currently running.\nAborting might leave incomplete files.\n\nExit anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.update_status("*** Exit requested during operation - Attempting cancellation ***")
                if self.active_worker:
                    self.active_worker.is_cancelled = True # Signal worker
                # Give thread a moment to potentially react/clean up, then accept close
                # Don't wait indefinitely here, main thread needs to exit
                event.accept()
            else:
                event.ignore() # Don't close
        else:
            event.accept() # Close normally


    def on_mount_checkbox_changed(self, state):
        """Handle mount checkbox state changes."""
        self.update_status(f"Mount after creation: {'Enabled' if state == Qt.CheckState.Checked else 'Disabled'}")


# --- Main Execution ---
if __name__ == '__main__':
    # Enable high DPI scaling for better look on modern displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    main_window = VHDXCreatorApp() # Styles are applied inside __init__ now
    main_window.show()

    sys.exit(app.exec())