# PyClipManager - Advanced Python Clipboard Manager (Single File)
# Version: 0.8 (Demo)

import sys
import os
import sqlite3
import datetime
import io
import platform
import traceback # For better error logging

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QListWidget, QListWidgetItem, QTextEdit, QLabel,
    QSystemTrayIcon, QMenu, QMessageBox, QPushButton, QSplitter,
    QSizePolicy
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QAction, QClipboard, QImageReader, QKeySequence,
    QPainter, QColor, QFont # Added QFont
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QByteArray, QMimeData, QUrl,
    QSettings, QSize, QBuffer, QIODevice, pyqtSlot # Added pyqtSlot
)

# --- Other Libraries ---
try:
    from pynput import keyboard
except ImportError:
    print("Error: pynput library not found. Global hotkey will not work.")
    print("Install using: pip install pynput")
    keyboard = None # Flag that pynput is missing

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library not found. Image handling will be limited.")
    print("Install using: pip install Pillow")
    Image = None # Flag that Pillow is missing

try:
    import qtawesome
except ImportError:
    print("Warning: qtawesome library not found. Icons may be missing.")
    print("Install using: pip install qtawesome")
    qtawesome = None # Flag that qtawesome is missing

# --- Constants ---
APP_NAME = "PyClipManager"
DB_NAME = "clipboard_history.sqlite"
MAX_HISTORY_DEFAULT = 200
DEFAULT_HOTKEY = "<ctrl>+<alt>+v" # Use pynput format
SETTINGS_ORG = "MyCompany" # Change as needed
SETTINGS_APP = APP_NAME

# --- Helper Functions ---

def get_icon(name, default_icon=None):
    """Gets an icon using qtawesome or returns a default."""
    if qtawesome:
        try:
            return qtawesome.icon(name, color='white') # Assuming dark theme
        except Exception:
            print(f"Warning: Could not load qtawesome icon '{name}'")
            return default_icon or QIcon()
    return default_icon or QIcon()

def create_thumbnail(image_data, size=(64, 64)):
    """Creates a thumbnail from image byte data using Pillow."""
    if not Image:
        return None
    try:
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(size)
        # Convert back to bytes (PNG format for consistency)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return None

def format_timestamp(dt_str):
    """Formats a datetime string nicely."""
    try:
        dt = datetime.datetime.fromisoformat(dt_str)
        now = datetime.datetime.now()
        delta = now - dt
        if delta.days == 0:
            if delta.seconds < 60:
                return "Just now"
            elif delta.seconds < 3600:
                return f"{delta.seconds // 60} min ago"
            else:
                return dt.strftime("%H:%M") # Today's time
        elif delta.days == 1:
            return f"Yesterday {dt.strftime('%H:%M')}"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return dt_str # Return original if parsing fails

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # Access columns by name
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            # Consider raising an exception or handling more gracefully
            self.conn = None

    def _create_table(self):
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clipboard_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    content_type TEXT NOT NULL,
                    data BLOB NOT NULL,
                    preview TEXT,
                    search_text TEXT,
                    is_pinned INTEGER DEFAULT 0,
                    source_app TEXT NULLABLE
                )
            """)
            # Add indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON clipboard_history (timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_pinned ON clipboard_history (is_pinned);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_text ON clipboard_history (search_text);")
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database table creation error: {e}")

    def add_item(self, content_type, data, source_app=None):
        if not self.conn: return None
        preview_text = None
        search_text = None

        if content_type == 'text':
            text_data = data.decode('utf-8', errors='replace') # Decode bytes to string
            preview_text = (text_data[:100] + '...') if len(text_data) > 100 else text_data
            search_text = text_data.lower()
            db_data = data # Store original bytes
        elif content_type.startswith('image'):
            # Store image data as is (BLOB)
            db_data = data
            preview_text = "[Image]" # Simple preview text for images
        else:
            print(f"Unsupported content type for DB: {content_type}")
            return None # Or handle other types if needed

        sql = """
            INSERT INTO clipboard_history (content_type, data, preview, search_text, source_app)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (content_type, db_data, preview_text, search_text, source_app))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Database insert error: {e}")
            return None

    def get_items(self, search_term=None, limit=100, pinned_only=False, offset=0):
        if not self.conn: return []
        try:
            cursor = self.conn.cursor()
            base_sql = "SELECT id, timestamp, content_type, preview, is_pinned FROM clipboard_history"
            conditions = []
            params = []

            if pinned_only:
                 conditions.append("is_pinned = 1")
            elif search_term:
                conditions.append("(search_text LIKE ? OR content_type LIKE ?)")
                term = f"%{search_term.lower()}%"
                params.extend([term, term])

            if conditions:
                base_sql += " WHERE " + " AND ".join(conditions)

            # Always show pinned items first, then by time
            base_sql += " ORDER BY is_pinned DESC, timestamp DESC"
            base_sql += f" LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(base_sql, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database query error: {e}")
            return []

    def get_item_data(self, item_id):
        if not self.conn: return None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data, content_type FROM clipboard_history WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            return row if row else None
        except sqlite3.Error as e:
            print(f"Database fetch data error: {e}")
            return None

    def delete_item(self, item_id):
        if not self.conn: return False
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM clipboard_history WHERE id = ? AND is_pinned = 0", (item_id,)) # Only delete if not pinned
            self.conn.commit()
            return cursor.rowcount > 0 # Return True if a row was deleted
        except sqlite3.Error as e:
            print(f"Database delete error: {e}")
            return False

    def force_delete_item(self, item_id):
        """ Deletes an item regardless of pinned status """
        if not self.conn: return False
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM clipboard_history WHERE id = ?", (item_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database force delete error: {e}")
            return False


    def toggle_pin(self, item_id):
        if not self.conn: return False
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE clipboard_history SET is_pinned = 1 - is_pinned WHERE id = ?", (item_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database pin toggle error: {e}")
            return False

    def clear_history(self):
        """Deletes all non-pinned items."""
        if not self.conn: return False
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM clipboard_history WHERE is_pinned = 0")
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database clear history error: {e}")
            return False

    def prune_history(self, max_items):
        """Keeps only the most recent `max_items` non-pinned entries."""
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            # Find the timestamp of the Nth most recent non-pinned item
            cursor.execute("""
                SELECT timestamp FROM clipboard_history
                WHERE is_pinned = 0
                ORDER BY timestamp DESC
                LIMIT 1 OFFSET ?
            """, (max_items - 1,)) # -1 because offset is 0-based
            cutoff_row = cursor.fetchone()

            if cutoff_row:
                cutoff_timestamp = cutoff_row['timestamp']
                # Delete items older than the cutoff timestamp that are not pinned
                cursor.execute("""
                    DELETE FROM clipboard_history
                    WHERE is_pinned = 0 AND timestamp < ?
                """, (cutoff_timestamp,))
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database prune error: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

# --- Clipboard Monitor Thread ---
class ClipboardMonitor(QThread):
    clipboard_changed = pyqtSignal(str, object) # content_type, data (bytes)
    error_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clipboard = QApplication.clipboard()
        self.monitoring = True
        self.last_mime_data_sig = None # Store a signature to detect actual changes

    def _get_mime_data_signature(self, mime_data):
        """Generate a simple signature for MimeData to detect changes."""
        if not mime_data:
            return None
        formats = sorted(mime_data.formats())
        sig_parts = []
        if mime_data.hasText():
            sig_parts.append(f"text:{hash(mime_data.text())}")
        if mime_data.hasImage():
            # Use image data hash - might be slow for large images
            img_data = mime_data.imageData()
            if img_data:
                 # Convert QVariant (imageData) to QImage or QPixmap then bytes
                try:
                    qimg = QImageReader(QBuffer(img_data)).read() # Try reading as image
                    if not qimg.isNull():
                        ba = QByteArray()
                        buf = QBuffer(ba)
                        buf.open(QIODevice.OpenModeFlag.WriteOnly)
                        qimg.save(buf, "PNG") # Save as PNG for consistent hashing
                        sig_parts.append(f"image:{hash(ba.data())}")
                    else: # If not image, maybe handle other data types or just use format list
                       sig_parts.append(f"formats:{','.join(formats)}")
                except Exception: # Handle cases where imageData isn't a standard image
                    sig_parts.append(f"formats:{','.join(formats)}")
            else:
                 sig_parts.append(f"formats:{','.join(formats)}")

        elif mime_data.hasUrls():
            sig_parts.append(f"urls:{';'.join([url.toString() for url in mime_data.urls()])}")
        else: # Fallback for other types
             sig_parts.append(f"formats:{','.join(formats)}")

        return "|".join(sig_parts)

    @pyqtSlot()
    def check_clipboard(self):
        if not self.monitoring:
            return

        try:
            mime_data = self.clipboard.mimeData()
            current_sig = self._get_mime_data_signature(mime_data)

            # Only proceed if the signature has actually changed
            if mime_data and current_sig != self.last_mime_data_sig:
                self.last_mime_data_sig = current_sig
                content_type = None
                data = None

                # Prioritize Image > Text > URLs
                if mime_data.hasImage():
                    image_data_variant = mime_data.imageData()
                    if image_data_variant:
                        # Convert QVariant to QPixmap/QImage then bytes (PNG)
                        qimg = QImageReader(QBuffer(image_data_variant)).read()
                        if not qimg.isNull():
                            ba = QByteArray()
                            buffer = QBuffer(ba)
                            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                            qimg.save(buffer, "PNG") # Standardize to PNG
                            data = ba.data() # Get bytes
                            content_type = "image_png"
                        else:
                             print("Clipboard claims image, but data is invalid.")

                elif mime_data.hasText():
                    text = mime_data.text()
                    if text: # Ensure non-empty text
                        content_type = "text"
                        data = text.encode('utf-8') # Store as bytes

                elif mime_data.hasUrls():
                     urls = [url.toLocalFile() if url.isLocalFile() else url.toString() for url in mime_data.urls()]
                     if urls:
                        content_type = "text" # Treat file paths/URLs as text for now
                        data = "\n".join(urls).encode('utf-8')

                # --- Add other format checks here if needed ---

                if content_type and data:
                    # print(f"Clipboard changed: {content_type}") # Debug
                    self.clipboard_changed.emit(content_type, data)
                # else:
                #     print("Clipboard changed, but no supported data detected.") # Debug

        except Exception as e:
            error_msg = f"Error reading clipboard: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_signal.emit(error_msg) # Emit error signal


    def run(self):
        # Initial check
        self.check_clipboard()
        # Connect signal for subsequent changes
        try:
            self.clipboard.dataChanged.connect(self.check_clipboard)
            self.exec() # Start thread event loop to keep signal connection alive
        except Exception as e:
             error_msg = f"Failed to connect clipboard signal: {e}"
             print(error_msg)
             self.error_signal.emit(error_msg)

    def stop(self):
        self.monitoring = False
        try:
            self.clipboard.dataChanged.disconnect(self.check_clipboard)
        except TypeError: # Signal might not be connected
            pass
        except Exception as e:
            print(f"Error disconnecting clipboard signal: {e}")
        self.quit()
        self.wait(2000) # Wait max 2 seconds for thread to finish

    def toggle_monitoring(self):
        self.monitoring = not self.monitoring
        status = "Resumed" if self.monitoring else "Paused"
        print(f"Clipboard monitoring {status}")
        if self.monitoring:
            self.check_clipboard() # Check immediately when resuming
        return self.monitoring


# --- Hotkey Listener Thread ---
class HotkeyListener(QThread):
    hotkey_pressed = pyqtSignal()

    def __init__(self, hotkey_str, parent=None):
        super().__init__(parent)
        self._hotkey_str = hotkey_str
        self._listener = None
        self._running = False

    def run(self):
        if not keyboard:
            print("Hotkey listener disabled: pynput not available.")
            return

        self._running = True
        try:
            # Define the callback for when the hotkey is pressed
            def on_activate():
                # Emit the signal from the main thread's context if possible,
                # but emitting directly often works fine with Qt signals.
                self.hotkey_pressed.emit()
                # print(f"Hotkey {self._hotkey_str} activated!") # Debug

            # Set up the listener with the specific hotkey combination
            # The format needs to match pynput's GlobalHotKeys expectation
            # e.g., '<ctrl>+<alt>+v'
            hotkey_map = {self._hotkey_str: on_activate}
            self._listener = keyboard.GlobalHotKeys(hotkey_map)

            print(f"Starting hotkey listener for: {self._hotkey_str}")
            # Start listening; this call blocks until listener.stop() is called
            self._listener.start() # Start in this thread
            self._listener.join()  # Wait for listener thread to finish (on stop)

        except ValueError as ve:
             print(f"Error setting up hotkey '{self._hotkey_str}': {ve}. Is the format correct (e.g., <ctrl>+<alt>+v)?")
        except Exception as e:
            # This might catch OS-level permission errors too
            print(f"Hotkey listener error: {e}")
            print("Ensure the application has necessary permissions (e.g., Accessibility on macOS).")
            traceback.print_exc()
        finally:
            self._running = False
            print("Hotkey listener stopped.")


    def stop(self):
        if self._listener and self._running:
            print("Stopping hotkey listener...")
            try:
                self._listener.stop()
            except Exception as e:
                print(f"Error stopping listener: {e}")
        self.quit() # Quit QThread event loop if it was used
        self.wait(1000) # Wait briefly


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self, db_manager, settings):
        super().__init__()
        self.db_manager = db_manager
        self.settings = settings
        self.max_history = int(self.settings.value("max_history", MAX_HISTORY_DEFAULT))
        self.current_hotkey = self.settings.value("global_hotkey", DEFAULT_HOTKEY)
        self.is_monitoring = True

        # --- Data ---
        self.item_cache = {} # Cache full data for preview {item_id: (type, data)}

        self.init_ui()
        self.load_history()

        # --- Threads ---
        self.monitor_thread = ClipboardMonitor()
        self.monitor_thread.clipboard_changed.connect(self.add_history_item)
        self.monitor_thread.error_signal.connect(self.show_error_message)
        self.monitor_thread.start()

        self.hotkey_thread = None
        self.setup_hotkey_listener()


    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        # self.setWindowIcon(get_icon("fa5s.clipboard-list", QIcon("path/to/default/icon.png"))) # Set proper icon path if needed

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduce margins
        main_layout.setSpacing(5)

        # --- Search Bar ---
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search history...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.filter_history)
        main_layout.addWidget(self.search_input)

        # --- Splitter for List and Preview ---
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self.splitter)

        # --- History List ---
        self.history_list = QListWidget()
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_item_context_menu)
        self.history_list.itemClicked.connect(self.show_preview)
        self.history_list.itemDoubleClicked.connect(self.copy_selected_to_clipboard)
        # Make list take available space initially
        self.history_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(self.history_list)


        # --- Preview Area ---
        self.preview_area = QWidget()
        preview_layout = QVBoxLayout(self.preview_area)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Select an item to preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setWordWrap(True)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True) # Initially read-only
        self.preview_text.setVisible(False) # Hide one initially
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.preview_text)
        # Make preview smaller initially
        self.preview_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.preview_area)
        self.splitter.setSizes([400, 150]) # Initial size ratio


        # --- System Tray Icon ---
        self.tray_icon = QSystemTrayIcon(self)
        # self.tray_icon.setIcon(self.windowIcon()) # Use window icon or specific tray icon
        self.tray_icon.setIcon(get_icon("fa5s.clipboard", QIcon())) # Placeholder
        self.tray_icon.setToolTip(APP_NAME)

        tray_menu = QMenu()
        show_action = QAction("Show", self, triggered=self.show_window)
        hide_action = QAction("Hide", self, triggered=self.hide_window) # Explicit hide action
        self.monitor_action = QAction("Pause Monitoring", self, triggered=self.toggle_monitoring, checkable=True)
        self.monitor_action.setChecked(False) # Start in paused state? No, start checked == monitoring
        clear_action = QAction("Clear History (Non-Pinned)", self, triggered=self.clear_history)
        # Settings action needed here
        quit_action = QAction("Quit", self, triggered=self.quit_app)

        tray_menu.addAction(show_action)
        tray_menu.addAction(hide_action)
        tray_menu.addSeparator()
        tray_menu.addAction(self.monitor_action)
        tray_menu.addAction(clear_action)
        # Add settings action here
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.handle_tray_activation)
        self.tray_icon.show()

        # Apply Dark Theme (ensure qss file exists or define style here)
        self.apply_dark_theme()

        self.resize(500, 600) # Default size

    def apply_dark_theme(self):
        # Basic Dark Theme QSS (customize as needed)
        dark_qss = """
            QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
                border: none;
                font-size: 10pt; /* Adjust font size */
            }
            QMainWindow {
                border: 1px solid #1e1e1e;
            }
            QLineEdit {
                background-color: #3c3f41;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #3c3f41;
                border: 1px solid #555;
                padding: 2px; /* Reduce padding */
                alternate-background-color: #45494c; /* Subtle row difference */
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 5px 3px; /* Adjust item padding (V H) */
                border-bottom: 1px solid #4a4a4a; /* Separator line */
            }
            QListWidget::item:selected {
                background-color: #5a7b9c; /* Selection color */
                color: #ffffff;
            }
             QListWidget::item:hover {
                background-color: #4f5356; /* Hover color */
            }
            QTextEdit {
                background-color: #3c3f41;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QLabel#PreviewLabel { /* If you set object name */
                font-style: italic;
                color: #aaaaaa;
            }
            QSplitter::handle {
                background-color: #4a4a4a;
                height: 3px; /* Vertical splitter handle */
                width: 3px; /* Horizontal splitter handle */
            }
            QSplitter::handle:hover {
                background-color: #5a7b9c;
            }
            QSplitter::handle:pressed {
                background-color: #555;
            }
            QMenuBar {
                background-color: #3c3f41;
            }
            QMenuBar::item:selected {
                background-color: #5a7b9c;
            }
            QMenu {
                background-color: #3c3f41;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #5a7b9c;
            }
             QSystemTrayIcon::message { /* Style tray message if possible */
                color: #f0f0f0;
                background-color: #3c3f41;
                border: 1px solid #555;
             }
             QPushButton { /* Basic button styling */
                background-color: #555;
                border: 1px solid #666;
                padding: 5px 10px;
                border-radius: 3px;
                min-width: 60px;
             }
             QPushButton:hover {
                background-color: #666;
             }
             QPushButton:pressed {
                 background-color: #444;
             }

        """
        self.setStyleSheet(dark_qss)
        # Ensure specific widgets like preview labels/text inherit or have styles set
        self.preview_text.setStyleSheet("background-color: #333; color: #eee;") # Example override
        self.preview_label.setStyleSheet("color: #ccc;")


    def setup_hotkey_listener(self):
        if self.hotkey_thread and self.hotkey_thread.isRunning():
            self.hotkey_thread.stop() # Stop previous listener if exists

        if keyboard: # Only if pynput is available
            self.hotkey_thread = HotkeyListener(self.current_hotkey)
            self.hotkey_thread.hotkey_pressed.connect(self.toggle_window_visibility)
            self.hotkey_thread.start()
        else:
            self.show_error_message("Global hotkey disabled (pynput library missing or failed to load).")


    def load_history(self, search_term=None):
        self.history_list.clear()
        self.item_cache.clear() # Clear cache when reloading

        # Get pinned items first if not searching
        items = []
        if not search_term:
           pinned_items = self.db_manager.get_items(pinned_only=True, limit=self.max_history)
           items.extend(pinned_items)

        # Get remaining/searched items
        remaining_limit = self.max_history - len(items) if not search_term else self.max_history
        if remaining_limit > 0:
            # If searching, get all matching, otherwise get recent non-pinned
            other_items = self.db_manager.get_items(search_term=search_term, limit=remaining_limit)
            # Avoid duplicates if searching included pinned items already fetched
            if search_term:
                 items = other_items
            else:
                 # Add non-pinned items only if they weren't already added as pinned
                 pinned_ids = {item['id'] for item in items}
                 items.extend([item for item in other_items if item['id'] not in pinned_ids])


        # Populate List Widget
        for item in items:
            list_item = QListWidgetItem()
            item_id = item['id']
            content_type = item['content_type']
            preview_text = item['preview'] if item['preview'] else ""
            timestamp = format_timestamp(item['timestamp'])
            is_pinned = item['is_pinned']

            icon = QIcon()
            if content_type == 'text':
                icon = get_icon("fa5s.file-alt", QIcon()) # Text icon
            elif content_type.startswith('image'):
                icon = get_icon("fa5s.image", QIcon()) # Image icon

            pin_char = "📌 " if is_pinned else ""
            display_text = f"{pin_char}{preview_text}\n<small style='color:#aaa'>{timestamp}</small>" # Simple HTML for timestamp style

            # Use QLabel for rich text rendering in list item
            item_widget = QLabel(display_text)
            item_widget.setTextFormat(Qt.TextFormat.RichText) # Enable HTML subset
            item_widget.setWordWrap(True)

            # Set fixed height or manage size hint if needed for QLabel wrapping
            # list_item.setSizeHint(item_widget.sizeHint()) # May need adjustment

            list_item.setIcon(icon)
            list_item.setData(Qt.ItemDataRole.UserRole, item_id) # Store DB id
            list_item.setData(Qt.ItemDataRole.UserRole + 1, content_type) # Store type

            # Set the custom widget for the item
            list_item.setSizeHint(item_widget.sizeHint() + QSize(0, 5)) # Add padding
            self.history_list.addItem(list_item)
            self.history_list.setItemWidget(list_item, item_widget)


    def filter_history(self):
        search_term = self.search_input.text().strip()
        self.load_history(search_term=search_term if search_term else None)

    @pyqtSlot(str, object) # content_type, data (bytes)
    def add_history_item(self, content_type, data):
        # Add to DB
        new_id = self.db_manager.add_item(content_type, data)
        if new_id:
            # Prune old items after adding
            self.db_manager.prune_history(self.max_history)
            # Refresh list (could optimize by prepending)
            self.load_history(search_term=self.search_input.text().strip() or None)
            # Optionally notify user
            self.tray_icon.showMessage(APP_NAME, f"{content_type.capitalize()} item copied.", self.tray_icon.icon(), 1500)


    def show_preview(self, item):
        if not item:
            # Clear preview if no item selected
            self.preview_text.setVisible(False)
            self.preview_label.setText("Select an item to preview")
            self.preview_label.setVisible(True)
            self.preview_label.setPixmap(QPixmap()) # Clear image
            self.preview_text.setReadOnly(True) # Reset edit state
            return

        item_id = item.data(Qt.ItemDataRole.UserRole)
        content_type = item.data(Qt.ItemDataRole.UserRole + 1)

        # Check cache first
        cached_data = self.item_cache.get(item_id)
        if cached_data:
            c_type, full_data = cached_data
        else:
            # Fetch from DB if not cached
            db_result = self.db_manager.get_item_data(item_id)
            if not db_result:
                self.preview_label.setText("Error retrieving data.")
                self.preview_label.setVisible(True)
                self.preview_text.setVisible(False)
                return
            full_data, c_type = db_result # Unpack directly
            content_type = c_type # Use type from DB
            self.item_cache[item_id] = (c_type, full_data) # Cache it

        self.preview_text.setVisible(False)
        self.preview_label.setVisible(False) # Hide both initially

        if content_type == 'text':
            try:
                text_content = full_data.decode('utf-8')
                self.preview_text.setPlainText(text_content)
                self.preview_text.setVisible(True)
                 # Allow editing for text items
                self.preview_text.setReadOnly(False)
                self.preview_text.textChanged.connect(lambda: self.mark_preview_edited(item_id)) # Mark as edited on change
            except UnicodeDecodeError:
                self.preview_label.setText("[Error decoding text]")
                self.preview_label.setVisible(True)
                self.preview_text.setReadOnly(True) # Cannot edit if decode fails

        elif content_type.startswith('image'):
            self.preview_text.setReadOnly(True) # Cannot edit images here
            pixmap = QPixmap()
            if pixmap.loadFromData(full_data):
                 # Scale pixmap to fit the label width, maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(self.preview_area.width() - 20, # Adjust margin
                                                    Qt.TransformationMode.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setText("[Invalid Image Data]")
            self.preview_label.setVisible(True)
        else:
            self.preview_text.setReadOnly(True)
            self.preview_label.setText(f"[Unsupported Type: {content_type}]")
            self.preview_label.setVisible(True)

    def mark_preview_edited(self, item_id):
        # Placeholder: Could visually indicate the preview has changed
        # print(f"Preview for item {item_id} edited.")
        pass # Currently no visual indication needed, direct copy works

    def copy_selected_to_clipboard(self, item=None):
        if item is None:
            item = self.history_list.currentItem()
        if not item:
            return

        item_id = item.data(Qt.ItemDataRole.UserRole)
        content_type = item.data(Qt.ItemDataRole.UserRole + 1)

        # Check if preview text was edited
        if content_type == 'text' and self.preview_text.isVisible() and not self.preview_text.isReadOnly():
            current_preview_text = self.preview_text.toPlainText()
            # Compare with cached/original data if necessary, or just use preview
            print("Copying edited text from preview.")
            mime_data = QMimeData()
            mime_data.setText(current_preview_text)
            QApplication.clipboard().setMimeData(mime_data)
            self.flash_item(item) # Visual feedback
            return

        # If not edited text, use cached/original data
        cached_data = self.item_cache.get(item_id)
        if not cached_data:
            db_result = self.db_manager.get_item_data(item_id)
            if not db_result:
                self.show_error_message("Failed to retrieve item data to copy.")
                return
            full_data, db_content_type = db_result
            content_type = db_content_type # Ensure we use the correct type from DB
        else:
            _, full_data = cached_data # Type is already known

        mime_data = QMimeData()
        if content_type == 'text':
            try:
                mime_data.setText(full_data.decode('utf-8'))
            except UnicodeDecodeError:
                 self.show_error_message("Cannot copy: Text data is corrupted.")
                 return
        elif content_type.startswith('image'):
             # Load bytes into QPixmap/QImage to put on clipboard correctly
             qimg = QImageReader(QBuffer(QByteArray(full_data))).read()
             if not qimg.isNull():
                mime_data.setImageData(qimg) # Set image data
             else:
                self.show_error_message("Cannot copy: Image data is invalid.")
                return
        else:
             self.show_error_message(f"Cannot copy unsupported type: {content_type}")
             return

        QApplication.clipboard().setMimeData(mime_data)
        print(f"Item {item_id} ({content_type}) copied to clipboard.")
        self.flash_item(item) # Visual feedback
        self.tray_icon.showMessage(APP_NAME, f"Item copied to clipboard.", self.tray_icon.icon(), 1000)
        # Option: Hide window after copy?
        # self.hide_window()


    def show_item_context_menu(self, pos):
        item = self.history_list.itemAt(pos)
        if not item:
            return

        item_id = item.data(Qt.ItemDataRole.UserRole)
        content_type = item.data(Qt.ItemDataRole.UserRole + 1)
        db_item_info = self.db_manager.conn.cursor().execute("SELECT is_pinned FROM clipboard_history WHERE id = ?", (item_id,)).fetchone()
        is_pinned = db_item_info['is_pinned'] if db_item_info else False

        context_menu = QMenu(self)
        copy_action = context_menu.addAction(get_icon("fa5s.copy"), "Copy")
        copy_action.triggered.connect(lambda: self.copy_selected_to_clipboard(item))

        if content_type == 'text':
             copy_plain_action = context_menu.addAction(get_icon("fa5s.file-alt"), "Copy as Plain Text") # Same as copy for now
             copy_plain_action.triggered.connect(lambda: self.copy_selected_to_clipboard(item)) # Future: strip formatting if needed

        pin_text = "Unpin Item" if is_pinned else "Pin Item"
        pin_icon = get_icon("fa5s.thumbtack", QIcon()) # Use a pin icon
        pin_action = context_menu.addAction(pin_icon, pin_text)
        pin_action.triggered.connect(lambda: self.toggle_pin_item(item_id))

        context_menu.addSeparator()
        delete_action = context_menu.addAction(get_icon("fa5s.trash-alt"), "Delete Item")
        delete_action.triggered.connect(lambda: self.delete_item(item_id, force=is_pinned)) # Offer force delete for pinned

        context_menu.exec(self.history_list.mapToGlobal(pos))

    def toggle_pin_item(self, item_id):
        if self.db_manager.toggle_pin(item_id):
            self.load_history(search_term=self.search_input.text().strip() or None) # Reload to show pin status change
        else:
            self.show_error_message("Failed to toggle pin status.")

    def delete_item(self, item_id, force=False):
        if force:
            confirm = QMessageBox.question(self, "Confirm Delete",
                                           "This item is pinned. Are you sure you want to permanently delete it?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                           QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.No:
                return
            deleted = self.db_manager.force_delete_item(item_id)
        else:
            deleted = self.db_manager.delete_item(item_id)

        if deleted:
            self.item_cache.pop(item_id, None) # Remove from cache
            # Find and remove item from list widget directly (more efficient)
            for i in range(self.history_list.count()):
                list_item = self.history_list.item(i)
                if list_item and list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                    self.history_list.takeItem(i)
                    break
            # Clear preview if the deleted item was selected
            if self.history_list.currentItem() is None or self.history_list.currentItem().data(Qt.ItemDataRole.UserRole) == item_id:
                 self.show_preview(None)
        else:
            self.show_error_message("Failed to delete item (maybe it's pinned?).")


    def clear_history(self):
        confirm = QMessageBox.question(self, "Confirm Clear History",
                                       "Are you sure you want to delete all non-pinned items?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            if self.db_manager.clear_history():
                self.load_history() # Reload the list
                self.show_preview(None) # Clear preview
                self.tray_icon.showMessage(APP_NAME, "Clipboard history cleared.", self.tray_icon.icon(), 1500)
            else:
                self.show_error_message("Failed to clear history.")

    def toggle_monitoring(self):
        self.is_monitoring = self.monitor_thread.toggle_monitoring()
        self.monitor_action.setChecked(not self.is_monitoring) # Check state indicates paused
        self.monitor_action.setText("Resume Monitoring" if not self.is_monitoring else "Pause Monitoring")
        status = "resumed" if self.is_monitoring else "paused"
        self.tray_icon.showMessage(APP_NAME, f"Monitoring {status}.", self.tray_icon.icon(), 1000)

    def flash_item(self, item):
        """ Provide visual feedback on copy """
        if not item: return
        original_style = item.background()
        item.setBackground(QColor(80, 120, 160)) # Flash color
        QTimer.singleShot(200, lambda: item.setBackground(original_style) if item else None) # Reset color


    def handle_tray_activation(self, reason):
        # Show/hide on left-click, show menu on right-click (already handled by setContextMenu)
        if reason == QSystemTrayIcon.ActivationReason.Trigger: # Left-click
            self.toggle_window_visibility()

    def toggle_window_visibility(self):
        if self.isVisible() and self.isActiveWindow():
             self.hide_window()
        else:
            self.show_window()

    def show_window(self):
        self.show()
        self.activateWindow()
        self.raise_()
        # Reload history in case DB changed while hidden
        self.load_history(search_term=self.search_input.text().strip() or None)

    def hide_window(self):
         self.hide()


    @pyqtSlot(str)
    def show_error_message(self, message):
        """ Displays an error message to the user. """
        print(f"ERROR: {message}") # Log to console as well
        # Use tray message for less intrusive errors, MessageBox for critical ones
        if "clipboard" in message.lower() or "hotkey" in message.lower():
             self.tray_icon.showMessage(f"{APP_NAME} Error", message, QSystemTrayIcon.MessageIcon.Warning, 5000)
        else:
             QMessageBox.warning(self, f"{APP_NAME} Error", message)


    def closeEvent(self, event):
        # Hide to tray instead of quitting
        event.ignore()
        self.hide_window()
        self.tray_icon.showMessage(APP_NAME, "Running in background.", self.tray_icon.icon(), 1500)

    def quit_app(self):
        print("Quitting application...")
        self.settings.setValue("max_history", self.max_history)
        self.settings.setValue("global_hotkey", self.current_hotkey)

        if self.hotkey_thread and self.hotkey_thread.isRunning():
            self.hotkey_thread.stop()
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()

        self.db_manager.close()
        self.tray_icon.hide()
        QApplication.instance().quit()


# --- Main Execution ---
def main():
    app = QApplication(sys.argv)
    app.setOrganizationName(SETTINGS_ORG)
    app.setApplicationName(SETTINGS_APP)
    app.setQuitOnLastWindowClosed(False) # Keep running when main window closes

    # --- Settings ---
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

    # --- Database Setup ---
    # Get appropriate path for database file
    data_path = os.path.join(os.path.expanduser("~"), f".{APP_NAME.lower()}") # Hidden dir in home
    os.makedirs(data_path, exist_ok=True)
    db_file = os.path.join(data_path, DB_NAME)
    print(f"Using database: {db_file}")

    db_manager = DatabaseManager(db_file)
    if not db_manager.conn:
        QMessageBox.critical(None, "Database Error", f"Could not connect to database at {db_file}. Exiting.")
        sys.exit(1)

    # --- Main Window ---
    main_window = MainWindow(db_manager, settings)

    # Connect cleanup code to application exit signal
    app.aboutToQuit.connect(main_window.quit_app) # Ensure cleanup runs

    # Decide whether to show window initially or start minimized
    # For now, show it. Could add a setting later.
    main_window.show_window()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Ctrl+C detected, quitting.")
        main_window.quit_app()
    except Exception as e:
        print(f"Unhandled exception occurred: {e}")
        traceback.print_exc()
        main_window.quit_app() # Attempt cleanup
        sys.exit(1)


if __name__ == "__main__":
    main()