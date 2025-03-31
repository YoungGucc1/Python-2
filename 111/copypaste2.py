# --- START OF FILE copypaste.py ---

# Required Dependencies: PyQt6, pyperclip, pyautogui (for hotkey detection, not pasting)
# Optional: python-dateutil (for relative timestamps, not implemented here)
# Create requirements.txt:
# PyQt6>=6.4.0
# pyperclip>=1.8.0
# pyautogui>=0.9.50  # Used only for hotkey triggering, not pasting

import sys
import json
import os
import sqlite3
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QListWidget, QListWidgetItem, QPushButton, QTabWidget,
                            QLabel, QLineEdit, QMessageBox, QFileDialog,
                            QDialog, QFormLayout, QSizePolicy)
from PyQt6.QtGui import QIcon, QKeySequence, QColor, QFont, QFontMetrics, QShortcut
from PyQt6.QtCore import Qt, QSize, QTimer, QMimeData, pyqtSignal, QPropertyAnimation, QEasingCurve

import pyperclip
import pyautogui # Still needed for hotkey simulation trigger, BUT NOT FOR PASTING

# --- Constants ---
APP_NAME = "Modern Clipboard Manager"
DATA_DIR_NAME = "ClipboardManager" if sys.platform == "win32" else ".clipboardmanager"
DB_FILENAME = "clipboard.db"
HISTORY_LIMIT = 100
STATUS_TIMEOUT = 2500 # Milliseconds for status bar messages
COPY_FLASH_DURATION = 300 # Milliseconds for copy visual feedback

# --- Helper Functions ---
def get_data_dir():
    """Gets the application's data directory path."""
    if sys.platform == "win32":
        base_dir = os.getenv("APPDATA")
    else:
        base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, DATA_DIR_NAME)
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    except OSError as e:
        print(f"Error creating data directory {data_dir}: {e}", file=sys.stderr)
        # Fallback to current directory? Or raise? For now, let DB connection fail.
    return data_dir

# --- Database ---
class DatabaseManager:
    """Handles all SQLite database interactions."""
    def __init__(self):
        """Initializes the database path and ensures the table exists."""
        data_dir = get_data_dir()
        self.db_path = os.path.join(data_dir, DB_FILENAME)
        self.init_database()

    def _execute(self, query, params=(), fetch_one=False, fetch_all=False, commit=False):
        """Executes a query with error handling and connection management."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if commit:
                    conn.commit()
                if fetch_one:
                    return cursor.fetchone()
                if fetch_all:
                    return cursor.fetchall()
                return cursor.lastrowid if commit else None
        except sqlite3.Error as e:
            print(f"Database Error: {e}\nQuery: {query}\nParams: {params}", file=sys.stderr)
            # Consider raising a custom exception or returning a specific error indicator
            return None # Indicate error

    def init_database(self):
        """Creates the clipboard_items table if it doesn't exist."""
        query = '''
            CREATE TABLE IF NOT EXISTS clipboard_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                favorite BOOLEAN NOT NULL DEFAULT 0,
                hotkey TEXT UNIQUE -- Ensure hotkeys are unique
            )
        '''
        self._execute(query, commit=True)
        # Add index for faster favorite lookup
        index_query = "CREATE INDEX IF NOT EXISTS idx_favorite ON clipboard_items (favorite);"
        self._execute(index_query, commit=True)


    def add_item(self, text, timestamp=None, favorite=False, hotkey=None):
        """Adds a new item to the database."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = '''
            INSERT INTO clipboard_items (text, timestamp, favorite, hotkey)
            VALUES (?, ?, ?, ?)
        '''
        return self._execute(query, (text, timestamp, favorite, hotkey), commit=True)

    def get_all_items(self):
        """Retrieves all items, ordered by timestamp descending."""
        query = 'SELECT id, text, timestamp, favorite, hotkey FROM clipboard_items ORDER BY timestamp DESC'
        return self._execute(query, fetch_all=True) or [] # Return empty list on error

    def update_item(self, item_id, text=None, favorite=None, hotkey=None):
        """Updates specified fields of an item by its ID."""
        updates = []
        values = []
        if text is not None:
            updates.append("text = ?")
            values.append(text)
        if favorite is not None:
            updates.append("favorite = ?")
            values.append(favorite)
        # Allow setting hotkey to NULL explicitly
        if hotkey is not None or 'hotkey' in locals() and hotkey is None:
             updates.append("hotkey = ?")
             values.append(hotkey)

        if not updates:
            return True # No updates needed

        values.append(item_id)
        query = f'UPDATE clipboard_items SET {", ".join(updates)} WHERE id = ?'
        return self._execute(query, tuple(values), commit=True) is not None # Return success/fail

    def delete_item(self, item_id):
        """Deletes an item by its ID."""
        query = 'DELETE FROM clipboard_items WHERE id = ?'
        return self._execute(query, (item_id,), commit=True) is not None

    def clear_all(self):
        """Deletes all items from the database."""
        query = 'DELETE FROM clipboard_items'
        return self._execute(query, commit=True) is not None

    def get_favorites(self):
        """Retrieves all favorite items."""
        query = 'SELECT id, text, timestamp, favorite, hotkey FROM clipboard_items WHERE favorite = 1 ORDER BY timestamp DESC'
        return self._execute(query, fetch_all=True) or []

    def get_item_by_id(self, item_id):
        """Retrieves a single item by its ID."""
        query = 'SELECT id, text, timestamp, favorite, hotkey FROM clipboard_items WHERE id = ?'
        return self._execute(query, (item_id,), fetch_one=True)

    def get_item_by_hotkey(self, hotkey):
        """Retrieves a single item by its hotkey."""
        query = 'SELECT id, text, timestamp, favorite, hotkey FROM clipboard_items WHERE hotkey = ?'
        return self._execute(query, (hotkey,), fetch_one=True)

# --- Data Model ---
class ClipboardItem:
    """Represents a single clipboard entry."""
    def __init__(self, text, timestamp=None, favorite=False, item_id=None, hotkey=None):
        self.id = item_id
        self.text = text
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.favorite = bool(favorite)
        self.hotkey = hotkey

    def to_dict(self):
        """Serializes the item to a dictionary for JSON export."""
        return {
            "id": self.id, # Include ID for potential reference, though usually recreated on import
            "text": self.text,
            "timestamp": self.timestamp,
            "favorite": self.favorite,
            "hotkey": self.hotkey
        }

    @classmethod
    def from_db_tuple(cls, db_tuple):
        """Creates a ClipboardItem from a database row tuple."""
        if not db_tuple or len(db_tuple) < 5:
            return None
        return cls(item_id=db_tuple[0], text=db_tuple[1], timestamp=db_tuple[2],
                   favorite=bool(db_tuple[3]), hotkey=db_tuple[4])

    @classmethod
    def from_dict(cls, data):
        """Creates a ClipboardItem from a dictionary (e.g., during import)."""
        return cls(data["text"], data.get("timestamp"), data.get("favorite", False),
                   data.get("id"), data.get("hotkey")) # Allow ID from import? Maybe ignore.

# --- Custom Widgets ---
class HotkeyDialog(QDialog):
    """Dialog for capturing and assigning a keyboard shortcut."""
    def __init__(self, existing_hotkey=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Hotkey")
        self.setFixedSize(350, 150)
        self.setStyleSheet("""
            QDialog { background-color: #2C3E50; color: #ECF0F1; }
            QLabel { color: #ECF0F1; font-size: 14px; }
            QLineEdit { padding: 8px; border-radius: 4px; background-color: #34495E;
                        color: #ECF0F1; border: 1px solid #7F8C8D; font-size: 14px; }
            QPushButton { background-color: #3498DB; color: white; border-radius: 4px;
                          padding: 8px 16px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton#clearBtn { background-color: #E74C3C; }
            QPushButton#clearBtn:hover { background-color: #C0392B; }
        """)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.hotkey_edit = QLineEdit()
        self.hotkey_edit.setPlaceholderText("Press key combination (e.g., Ctrl+Alt+C)")
        self.hotkey_edit.setReadOnly(True)
        if existing_hotkey:
            self.hotkey_edit.setText(existing_hotkey)

        form_layout.addRow("Hotkey:", self.hotkey_edit)
        layout.addLayout(form_layout)

        info_label = QLabel("Press the desired key combination.\nModifiers (Ctrl, Alt, Shift) are required.")
        info_label.setStyleSheet("font-size: 11px; color: #BDC3C7;")
        layout.addWidget(info_label)

        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_hotkey)
        self.save_btn = QPushButton("Save")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.key_sequence = None
        if existing_hotkey:
            self.key_sequence = QKeySequence.fromString(existing_hotkey, QKeySequence.SequenceFormat.PortableText)


    def keyPressEvent(self, event):
        """Captures key presses to form the hotkey sequence."""
        key = event.key()
        modifiers = event.modifiers()

        # Ignore modifier-only presses or keys we don't want
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift, Qt.Key.Key_Meta,
                   Qt.Key.Key_unknown, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Allow clearing with Backspace/Delete if field has content
            if self.hotkey_edit.text() and key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                self.clear_hotkey()
                event.accept()
            else:
                 super().keyPressEvent(event)
            return

        # Require at least one modifier
        if not (modifiers & (Qt.KeyboardModifier.ControlModifier |
                             Qt.KeyboardModifier.AltModifier |
                             Qt.KeyboardModifier.ShiftModifier |
                             Qt.KeyboardModifier.MetaModifier)):
            self.hotkey_edit.setText("Modifier required!")
            QTimer.singleShot(1000, lambda: self.hotkey_edit.setText(self.key_sequence.toString(QKeySequence.SequenceFormat.PortableText) if self.key_sequence else ""))
            event.accept()
            return

        # Combine modifiers and key
        sequence = QKeySequence(modifiers | key)
        hotkey_text = sequence.toString(QKeySequence.SequenceFormat.PortableText) # e.g., "Ctrl+Alt+C"

        self.hotkey_edit.setText(hotkey_text)
        self.key_sequence = sequence
        event.accept()

    def get_hotkey_text(self):
        """Returns the captured hotkey as text, or None if cleared/invalid."""
        # Return None only if intentionally cleared, otherwise return current valid text
        return self.hotkey_edit.text() if self.key_sequence and self.hotkey_edit.text() else None

    def clear_hotkey(self):
        """Clears the current hotkey selection."""
        self.hotkey_edit.clear()
        self.key_sequence = None


class ClipboardItemWidget(QWidget):
    """Custom widget to display a clipboard item in the QListWidget."""
    def __init__(self, item: ClipboardItem, clipboard_manager, parent=None):
        super().__init__(parent)
        self.item = item
        self.clipboard_manager = clipboard_manager
        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        """Sets up the layout and child widgets."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Left side buttons (Favorite/Delete)
        left_buttons_layout = QVBoxLayout()
        left_buttons_layout.setSpacing(2)
        left_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.star_btn = QPushButton("★")
        self.star_btn.setFixedSize(28, 28)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setStyleSheet("""
            QPushButton { font-size: 16px; font-weight: bold; border-radius: 14px; padding: 0px; }
            QPushButton:hover { background-color: #2980B9; color: white; }
        """)
        self.star_btn.clicked.connect(self.toggle_favorite)
        left_buttons_layout.addWidget(self.star_btn)

        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(28, 28)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton { color: #E74C3C; font-size: 18px; font-weight: bold; border-radius: 14px; padding: 0px; }
            QPushButton:hover { background-color: #E74C3C; color: white; }
        """)
        self.delete_btn.clicked.connect(self.delete_item)
        left_buttons_layout.addWidget(self.delete_btn)

        layout.addLayout(left_buttons_layout)

        # Content Area (Text, Timestamp, Hotkey)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)

        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.text_label.setStyleSheet("font-size: 13px;")
        content_layout.addWidget(self.text_label)

        # Bottom row for timestamp and hotkey
        bottom_layout = QHBoxLayout()
        self.timestamp_label = QLabel()
        self.timestamp_label.setStyleSheet("font-size: 10px; color: #95A5A6;")
        bottom_layout.addWidget(self.timestamp_label)
        bottom_layout.addStretch()
        self.hotkey_label = QLabel()
        self.hotkey_label.setStyleSheet("font-size: 11px; color: #3498DB; font-weight: bold;")
        bottom_layout.addWidget(self.hotkey_label)

        content_layout.addLayout(bottom_layout)
        layout.addLayout(content_layout, 1) # Content stretches

    def update_display(self):
        """Updates the widget's content based on the ClipboardItem."""
        # Text (Truncated)
        max_chars = 150 # Show more initially
        display_text = self.item.text[:max_chars] + ("..." if len(self.item.text) > max_chars else "")
        display_text = display_text.replace("\n", " ↵ ") # Indicate newlines
        self.text_label.setText(display_text)
        self.text_label.setToolTip(self.item.text) # Show full text on hover

        # Timestamp
        self.timestamp_label.setText(self.item.timestamp)

        # Hotkey
        if self.item.hotkey:
            self.hotkey_label.setText(f"[{self.item.hotkey}]")
            self.hotkey_label.show()
        else:
            self.hotkey_label.hide()

        # Favorite Star
        if self.item.favorite:
            self.star_btn.setStyleSheet("""
                QPushButton { background-color: #F39C12; color: white; font-size: 16px;
                              font-weight: bold; border-radius: 14px; padding: 0px; }
                QPushButton:hover { background-color: #D35400; color: white; }
            """)
            self.star_btn.setToolTip("Remove from favorites")
        else:
            self.star_btn.setStyleSheet("""
                QPushButton { background-color: #34495E; color: #7F8C8D; font-size: 16px;
                              font-weight: bold; border-radius: 14px; padding: 0px; }
                QPushButton:hover { background-color: #2980B9; color: white; }
            """)
            self.star_btn.setToolTip("Add to favorites")

    def toggle_favorite(self):
        """Callback when the favorite button is clicked."""
        if self.clipboard_manager and self.item:
            self.clipboard_manager.toggle_favorite(self.item.id)

    def delete_item(self):
        """Callback when the delete button is clicked."""
        if self.clipboard_manager and self.item:
            self.clipboard_manager.remove_item(self.item.id)

# --- Main Application ---
class ClipboardManager(QMainWindow):
    """Main application window for the clipboard manager."""
    clipboard_changed_signal = pyqtSignal(str) # Signal for external clipboard changes

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(650, 550)
        self.clipboard_items = [] # In-memory list of ClipboardItem objects
        self.hotkeys = {} # Map hotkey string -> QShortcut
        self.db = DatabaseManager()
        self._last_copied_text = None # To prevent self-triggering
        self._ignore_clipboard_signals = False # Flag to temporarily ignore signals

        self.setup_theme()
        self.setup_ui()
        self.load_data()

        # Clipboard Monitoring (using a timer for polling robustness)
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.check_clipboard)
        self.monitor_timer.start(500) # Check every 500ms

        # Connect internal signal (used by check_clipboard)
        self.clipboard_changed_signal.connect(self.on_clipboard_content_changed)


    def setup_theme(self):
        """Applies the application-wide stylesheet."""
        self.setStyleSheet("""
            QMainWindow { background-color: #2C3E50; }
            QTabWidget::pane { border: 1px solid #405060; background-color: #34495E; border-radius: 4px; }
            QTabBar::tab { background-color: #34495E; color: #ECF0F1; border: 1px solid #405060;
                           border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
                           min-width: 10ex; padding: 10px 15px; margin-right: 1px; font-size: 14px; }
            QTabBar::tab:selected { background-color: #3498DB; border-color: #3498DB; }
            QTabBar::tab:hover { background-color: #405060; }
            QTabBar::tab:selected:hover { background-color: #2980B9; }
            QListWidget { background-color: #34495E; border-radius: 4px;
                          border: 1px solid #405060; color: #ECF0F1; font-size: 14px; padding: 5px; }
            QListWidget::item { border-radius: 4px; background-color: #2C3E50; margin: 4px 0px;
                                border: 1px solid #405060; /* Add subtle border */ }
            /* QListWidget::item:selected { background-color: #3498DB; border-color: #3498DB; } */ /* Selection handled by widget */
            /* QListWidget::item:hover { background-color: #405060; } */ /* Hover handled by widget */
            QPushButton { background-color: #3498DB; color: white; border-radius: 4px;
                          padding: 8px 16px; font-size: 14px; font-weight: bold; margin: 2px; }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #7F8C8D; }
            QLabel { color: #ECF0F1; font-size: 14px; }
            QLineEdit { padding: 8px; border-radius: 4px; background-color: #34495E;
                        color: #ECF0F1; border: 1px solid #7F8C8D; font-size: 13px; }
            QStatusBar { color: #ECF0F1; }
            QToolTip { background-color: #34495E; color: #ECF0F1; border: 1px solid #7F8C8D; padding: 5px; }
        """)

    def setup_ui(self):
        """Sets up the main UI layout and widgets."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.tab_widget = QTabWidget()

        # --- History Tab ---
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(5, 10, 5, 5)
        history_layout.setSpacing(5)

        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search History...")
        self.history_search.textChanged.connect(self.filter_history_list)
        history_layout.addWidget(self.history_search)

        self.history_list = QListWidget()
        self.history_list.setSpacing(5)
        self.history_list.setUniformItemSizes(False) # Needed for varying text heights
        self.history_list.itemActivated.connect(self.copy_selected_item) # Enter key copies
        self.history_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        history_layout.addWidget(self.history_list)

        self.history_placeholder = QLabel("Clipboard history is empty.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.history_placeholder.setStyleSheet("color: #95A5A6; font-style: italic;")
        history_layout.addWidget(self.history_placeholder)
        self.history_placeholder.hide() # Initially hidden

        # --- Favorites Tab ---
        favorites_tab = QWidget()
        favorites_layout = QVBoxLayout(favorites_tab)
        favorites_layout.setContentsMargins(5, 10, 5, 5)
        favorites_layout.setSpacing(5)

        self.fav_search = QLineEdit()
        self.fav_search.setPlaceholderText("Search Favorites...")
        self.fav_search.textChanged.connect(self.filter_favorites_list)
        favorites_layout.addWidget(self.fav_search)

        self.favorites_list = QListWidget()
        self.favorites_list.setSpacing(5)
        self.favorites_list.setUniformItemSizes(False)
        self.favorites_list.itemActivated.connect(self.copy_selected_item)
        self.favorites_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        favorites_layout.addWidget(self.favorites_list)

        self.fav_placeholder = QLabel("You haven't added any favorites yet.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.fav_placeholder.setStyleSheet("color: #95A5A6; font-style: italic;")
        favorites_layout.addWidget(self.fav_placeholder)
        self.fav_placeholder.hide() # Initially hidden

        fav_btn_layout = QHBoxLayout()
        self.copy_fav_btn = QPushButton("Copy")
        self.copy_fav_btn.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_fav_btn.clicked.connect(self.copy_selected_item)
        fav_btn_layout.addWidget(self.copy_fav_btn)

        self.hotkey_btn = QPushButton("Assign Hotkey")
        self.hotkey_btn.setIcon(QIcon.fromTheme("preferences-desktop-keyboard-shortcuts"))
        self.hotkey_btn.clicked.connect(self.assign_hotkey)
        fav_btn_layout.addWidget(self.hotkey_btn)
        favorites_layout.addLayout(fav_btn_layout)


        # --- Add Tabs ---
        self.tab_widget.addTab(history_tab, "History")
        self.tab_widget.addTab(favorites_tab, "Favorites")
        main_layout.addWidget(self.tab_widget)

        # --- Bottom Buttons (Import/Export/Clear) ---
        bottom_btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import")
        self.import_btn.setIcon(QIcon.fromTheme("document-open"))
        self.import_btn.clicked.connect(self.import_data)
        bottom_btn_layout.addWidget(self.import_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_btn.clicked.connect(self.export_data)
        bottom_btn_layout.addWidget(self.export_btn)

        bottom_btn_layout.addStretch() # Push clear button to the right

        self.clear_all_btn = QPushButton("Clear All History")
        self.clear_all_btn.setIcon(QIcon.fromTheme("edit-clear-all"))
        self.clear_all_btn.setStyleSheet("background-color: #E74C3C;")
        self.clear_all_btn.clicked.connect(self.clear_all_data)
        bottom_btn_layout.addWidget(self.clear_all_btn)

        main_layout.addLayout(bottom_btn_layout)

        # --- Status Bar ---
        self.statusBar().setStyleSheet("padding: 3px;") # Add some padding

    # --- Clipboard Monitoring ---
    def check_clipboard(self):
        """Polls the clipboard and emits a signal if text content has changed."""
        if self._ignore_clipboard_signals:
            return
        try:
            current_text = pyperclip.paste()
            # Check if it's text, not empty, and different from the last known *external* copy
            if isinstance(current_text, str) and current_text and current_text != self._last_copied_text:
                 # Check if it's different from *any* item already in our list
                 # (Avoid adding duplicates rapidly if copy events fire multiple times)
                 if not any(item.text == current_text for item in self.clipboard_items):
                    self._last_copied_text = current_text # Update last known external copy
                    self.clipboard_changed_signal.emit(current_text)

        except pyperclip.PyperclipException as e:
            # Can happen if clipboard is busy or contains non-text data
            # print(f"Could not read clipboard: {e}", file=sys.stderr)
            pass # Ignore benign errors
        except Exception as e: # Catch unexpected errors
             print(f"Unexpected error checking clipboard: {e}", file=sys.stderr)
             self._last_copied_text = None # Reset comparison on error


    def on_clipboard_content_changed(self, text):
        """Handles the clipboard change signal, adding the new item."""
        # print(f"Clipboard changed: {text[:30]}...") # Debug
        if text and text.strip():
            self.add_to_history(text)

    # --- Core Logic ---
    def add_to_history(self, text):
        """Adds text to history, moving duplicates to top, enforcing limit."""
        if self._ignore_clipboard_signals: # Should not happen if check_clipboard is robust, but safety first
             return

        existing_item = next((item for item in self.clipboard_items if item.text == text), None)

        if existing_item:
            # Move existing item to the top (most recent)
            self.clipboard_items.remove(existing_item)
            self.clipboard_items.insert(0, existing_item)
            # Update timestamp in DB? Optional, keeps original time if not updated.
            # self.db.update_item(existing_item.id, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Moved existing item to top: {text[:30]}...") # Debug
        else:
            # Add new item
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            item_id = self.db.add_item(text, timestamp)
            if item_id is None:
                self.statusBar().showMessage("Error adding item to database", STATUS_TIMEOUT)
                return

            new_item = ClipboardItem(text, timestamp, item_id=item_id)
            self.clipboard_items.insert(0, new_item)
            print(f"Added new item: {text[:30]}...") # Debug

            # Enforce history limit (remove oldest non-favorite)
            if len(self.clipboard_items) > HISTORY_LIMIT:
                # Find the oldest non-favorite item to remove
                item_to_remove = None
                for i in range(len(self.clipboard_items) - 1, -1, -1): # Iterate backwards
                    if not self.clipboard_items[i].favorite:
                        item_to_remove = self.clipboard_items[i]
                        break

                if item_to_remove:
                    if self.db.delete_item(item_to_remove.id):
                         self.clipboard_items.remove(item_to_remove)
                         print(f"Removed oldest non-favorite due to limit: {item_to_remove.text[:30]}...") # Debug
                    else:
                         self.statusBar().showMessage("Error removing oldest item from database", STATUS_TIMEOUT)
                # If all are favorites, don't remove anything? Or remove oldest favorite?
                # Current logic: only removes non-favorites.

        self.refresh_ui()


    def copy_selected_item(self):
        """Copies the text of the currently selected item in the active tab's list."""
        current_list = self.tab_widget.currentWidget().findChild(QListWidget)
        if not current_list: return

        selected_qt_item = current_list.currentItem()
        if selected_qt_item:
            item_id = selected_qt_item.data(Qt.ItemDataRole.UserRole)
            clip_item = self._find_item_by_id(item_id)
            if clip_item:
                self.copy_item_text(clip_item, selected_qt_item)


    def copy_item_text(self, clip_item: ClipboardItem, list_widget_item: QListWidgetItem = None):
        """Copies the given item's text to the system clipboard with feedback."""
        try:
            # Temporarily ignore clipboard signals to prevent self-triggering
            self._ignore_clipboard_signals = True
            pyperclip.copy(clip_item.text)
            self._last_copied_text = clip_item.text # Update internal state
            self.statusBar().showMessage(f"Copied: {clip_item.text[:50]}...", STATUS_TIMEOUT)

            # Visual feedback: Flash background of the QListWidgetItem
            if list_widget_item:
                 widget = self.history_list.itemWidget(list_widget_item) or self.favorites_list.itemWidget(list_widget_item)
                 if widget:
                     original_style = widget.styleSheet()
                     flash_style = original_style + " background-color: #3498DB;" # Use highlight color
                     widget.setStyleSheet(flash_style)
                     QTimer.singleShot(COPY_FLASH_DURATION, lambda w=widget, s=original_style: w.setStyleSheet(s) if w else None)

        except pyperclip.PyperclipException as e:
            self.statusBar().showMessage(f"Error copying to clipboard: {e}", STATUS_TIMEOUT)
            QMessageBox.warning(self, "Copy Error", f"Could not copy text to clipboard:\n{e}")
        except Exception as e:
            self.statusBar().showMessage(f"Unexpected copy error: {e}", STATUS_TIMEOUT)
            QMessageBox.critical(self, "Copy Error", f"An unexpected error occurred during copy:\n{e}")
        finally:
             # Crucial: Re-enable clipboard monitoring after a short delay
             QTimer.singleShot(100, lambda: setattr(self, '_ignore_clipboard_signals', False))


    def toggle_favorite(self, item_id):
        """Toggles the favorite status of an item by its ID."""
        item = self._find_item_by_id(item_id)
        if not item: return

        new_favorite_status = not item.favorite
        item.favorite = new_favorite_status

        # Update database
        success = self.db.update_item(item.id, favorite=new_favorite_status)

        if success:
            if new_favorite_status:
                self.statusBar().showMessage("Added to favorites", STATUS_TIMEOUT)
            else:
                # If removing from favorites, also remove its hotkey
                if item.hotkey:
                    self._remove_hotkey_binding(item.hotkey)
                    item.hotkey = None
                    # Update DB again to clear hotkey
                    self.db.update_item(item.id, hotkey=None)
                self.statusBar().showMessage("Removed from favorites", STATUS_TIMEOUT)
            self.refresh_ui()
        else:
             # Revert in-memory change if DB update failed
             item.favorite = not new_favorite_status
             self.statusBar().showMessage("Error updating favorite status in database", STATUS_TIMEOUT)


    def remove_item(self, item_id):
        """Removes an item completely from history and favorites by its ID."""
        item = self._find_item_by_id(item_id)
        if not item: return

        reply = QMessageBox.question(
            self, "Confirm Delete", f"Are you sure you want to permanently delete this item?\n\n{item.text[:100]}...",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove hotkey if assigned
            if item.hotkey:
                self._remove_hotkey_binding(item.hotkey)

            # Remove from database
            if self.db.delete_item(item.id):
                # Remove from memory
                self.clipboard_items.remove(item)
                self.refresh_ui()
                self.statusBar().showMessage("Item deleted", STATUS_TIMEOUT)
            else:
                self.statusBar().showMessage("Error deleting item from database", STATUS_TIMEOUT)


    def assign_hotkey(self):
        """Opens dialog to assign a hotkey to the selected favorite item."""
        selected_qt_item = self.favorites_list.currentItem()
        if not selected_qt_item:
            QMessageBox.warning(self, "Assign Hotkey", "Please select a favorite item first.")
            return

        item_id = selected_qt_item.data(Qt.ItemDataRole.UserRole)
        clip_item = self._find_item_by_id(item_id)
        if not clip_item: return # Should not happen

        dialog = HotkeyDialog(existing_hotkey=clip_item.hotkey, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_hotkey_text = dialog.get_hotkey_text() # Returns None if cleared

            # Check if the new hotkey is already in use (by a DIFFERENT item)
            existing_item_with_hotkey = self._find_item_by_hotkey(new_hotkey_text)
            if new_hotkey_text and existing_item_with_hotkey and existing_item_with_hotkey.id != clip_item.id:
                QMessageBox.warning(self, "Hotkey Conflict",
                                    f"Hotkey '{new_hotkey_text}' is already assigned to another item:\n"
                                    f"{existing_item_with_hotkey.text[:50]}...")
                return

            # Remove old hotkey binding if it exists
            if clip_item.hotkey:
                self._remove_hotkey_binding(clip_item.hotkey)

            # Update item and database
            clip_item.hotkey = new_hotkey_text
            if self.db.update_item(clip_item.id, hotkey=new_hotkey_text):
                if new_hotkey_text:
                    # Add new hotkey binding
                    self._add_hotkey_binding(clip_item)
                    self.statusBar().showMessage(f"Hotkey '{new_hotkey_text}' assigned", STATUS_TIMEOUT)
                else:
                    self.statusBar().showMessage("Hotkey cleared", STATUS_TIMEOUT)
                self.refresh_ui() # Update display
            else:
                 # Revert in-memory change if DB failed
                 # (Need to know the previous hotkey to potentially restore binding) - Complex, maybe just show error.
                 clip_item.hotkey = dialog.hotkey_edit.text() # Or restore original before dialog
                 self.statusBar().showMessage("Error updating hotkey in database", STATUS_TIMEOUT)
                 # Consider re-adding old binding if needed


    def _add_hotkey_binding(self, item: ClipboardItem):
        """Creates and registers a QShortcut for the item's hotkey."""
        if not item.hotkey or item.hotkey in self.hotkeys:
            return # Avoid duplicates or empty hotkeys
        try:
            sequence = QKeySequence.fromString(item.hotkey, QKeySequence.SequenceFormat.PortableText)
            if sequence.isEmpty():
                 print(f"Warning: Could not create valid QKeySequence from '{item.hotkey}'", file=sys.stderr)
                 return

            shortcut = QShortcut(sequence, self)
            # Use lambda to capture the correct item at definition time
            shortcut.activated.connect(lambda item_to_activate=item: self.activate_hotkey(item_to_activate))
            shortcut.setEnabled(True)
            self.hotkeys[item.hotkey] = shortcut
            print(f"Bound hotkey: {item.hotkey}") # Debug
        except Exception as e:
             print(f"Error binding hotkey {item.hotkey}: {e}", file=sys.stderr)


    def _remove_hotkey_binding(self, hotkey_text):
        """Disables and removes a QShortcut."""
        if hotkey_text and hotkey_text in self.hotkeys:
            try:
                shortcut = self.hotkeys.pop(hotkey_text)
                shortcut.setEnabled(False)
                shortcut.deleteLater() # Clean up the QObject
                print(f"Unbound hotkey: {hotkey_text}") # Debug
            except Exception as e:
                 print(f"Error removing hotkey binding {hotkey_text}: {e}", file=sys.stderr)


    def activate_hotkey(self, item: ClipboardItem):
        """Called when a registered hotkey is pressed. Copies text."""
        print(f"Hotkey activated for: {item.text[:30]}...") # Debug
        # Find the corresponding QListWidgetItem *if visible* for feedback
        qt_list_item = None
        for i in range(self.favorites_list.count()):
             wl_item = self.favorites_list.item(i)
             if wl_item.data(Qt.ItemDataRole.UserRole) == item.id:
                 qt_list_item = wl_item
                 break
        # Also check history list if needed, though hotkeys are usually for favorites
        if not qt_list_item:
             for i in range(self.history_list.count()):
                 wl_item = self.history_list.item(i)
                 if wl_item.data(Qt.ItemDataRole.UserRole) == item.id:
                     qt_list_item = wl_item
                     break

        self.copy_item_text(item, qt_list_item)
        # Bring window to front? Optional.
        # self.activateWindow()
        # self.raise_()

    # --- UI Refresh and Filtering ---
    def refresh_ui(self):
        """Refreshes both history and favorites lists based on self.clipboard_items."""
        # Store current selections to try and restore them
        current_history_id = self.history_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.history_list.currentItem() else None
        current_fav_id = self.favorites_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.favorites_list.currentItem() else None

        # --- Populate History List ---
        self.history_list.clear()
        history_items_added = 0
        for item in self.clipboard_items: # Already sorted newest first
            list_item = QListWidgetItem(self.history_list) # Set parent
            widget = ClipboardItemWidget(item, self) # Pass manager for callbacks

            list_item.setSizeHint(widget.sizeHint())
            list_item.setData(Qt.ItemDataRole.UserRole, item.id) # Store ID

            self.history_list.addItem(list_item)
            self.history_list.setItemWidget(list_item, widget)
            history_items_added += 1

            if item.id == current_history_id:
                self.history_list.setCurrentItem(list_item)

        # Show/hide placeholder
        if history_items_added == 0:
            self.history_list.hide()
            self.history_placeholder.show()
        else:
            self.history_list.show()
            self.history_placeholder.hide()

        # Apply current filter
        self.filter_history_list()


        # --- Populate Favorites List ---
        self.favorites_list.clear()
        fav_items_added = 0
        favorite_items = sorted([item for item in self.clipboard_items if item.favorite], key=lambda x: x.timestamp, reverse=True)

        for item in favorite_items:
            list_item = QListWidgetItem(self.favorites_list)
            widget = ClipboardItemWidget(item, self)

            list_item.setSizeHint(widget.sizeHint())
            list_item.setData(Qt.ItemDataRole.UserRole, item.id)

            self.favorites_list.addItem(list_item)
            self.favorites_list.setItemWidget(list_item, widget)
            fav_items_added +=1

            if item.id == current_fav_id:
                 self.favorites_list.setCurrentItem(list_item)

        # Show/hide placeholder
        if fav_items_added == 0:
            self.favorites_list.hide()
            self.fav_placeholder.show()
        else:
            self.favorites_list.show()
            self.fav_placeholder.hide()

        # Apply current filter
        self.filter_favorites_list()


    def filter_history_list(self):
        """Filters the history list based on the search query."""
        query = self.history_search.text().lower()
        self._filter_list(self.history_list, query)

    def filter_favorites_list(self):
        """Filters the favorites list based on the search query."""
        query = self.fav_search.text().lower()
        self._filter_list(self.favorites_list, query)

    def _filter_list(self, list_widget: QListWidget, query: str):
        """Helper function to hide/show items in a QListWidget based on query."""
        items_visible = 0
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if isinstance(widget, ClipboardItemWidget):
                item_text = widget.item.text.lower()
                hotkey_text = widget.item.hotkey.lower() if widget.item.hotkey else ""
                # Match query in text OR hotkey
                if query in item_text or (query and query in hotkey_text):
                    item.setHidden(False)
                    items_visible += 1
                else:
                    item.setHidden(True)

        # Show placeholder only if list *should* be visible but no items match filter
        placeholder = self.history_placeholder if list_widget == self.history_list else self.fav_placeholder
        has_items_in_backend = any(True for _ in range(list_widget.count())) # Check if list was ever populated

        if has_items_in_backend and items_visible == 0 and query:
            placeholder.setText(f"No items match '{query}'.")
            placeholder.show()
            list_widget.hide()
        elif has_items_in_backend:
             placeholder.hide()
             list_widget.show()
        # If no items in backend, refresh_ui handles placeholder visibility based on total count.



    # --- Data Persistence ---
    def load_data(self):
        """Loads clipboard items from the SQLite database."""
        print("Loading data from database...")
        self.clipboard_items = []
        self.hotkeys = {} # Clear existing shortcuts

        db_items = self.db.get_all_items()
        if db_items is None: # Indicates DB error during fetch
             QMessageBox.critical(self, "Load Error", "Failed to load data from the database. Check console for details.")
             return

        loaded_count = 0
        hotkey_errors = []
        for item_tuple in db_items:
            item = ClipboardItem.from_db_tuple(item_tuple)
            if item:
                self.clipboard_items.append(item)
                # Register hotkey if applicable
                if item.favorite and item.hotkey:
                    try:
                        self._add_hotkey_binding(item)
                    except Exception as e:
                        hotkey_errors.append(f" - '{item.hotkey}': {e}")
                loaded_count += 1

        self.refresh_ui()
        self.statusBar().showMessage(f"Loaded {loaded_count} items.", STATUS_TIMEOUT)
        if hotkey_errors:
             error_msg = "Failed to bind the following hotkeys on load:\n" + "\n".join(hotkey_errors)
             QMessageBox.warning(self, "Hotkey Load Warning", error_msg)
        print(f"Loading complete. {len(self.hotkeys)} hotkeys active.")


    def import_data(self):
        """Imports clipboard items from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data", "", "JSON Files (*.json)"
        )
        if not file_path: return

        try:
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Import Error", f"Failed to parse JSON file: {e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read file: {e}")
            return

        imported_raw_items = data.get("clipboard_items", [])
        if not imported_raw_items:
            QMessageBox.information(self, "Import", "The selected file contains no clipboard items.")
            return

        # --- Import Options ---
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("Import Options")
        msgBox.setText(f"Found {len(imported_raw_items)} items to import. How to proceed?")
        msgBox.setIcon(QMessageBox.Icon.Question)
        replace_btn = msgBox.addButton("Replace Existing Data", QMessageBox.ButtonRole.YesRole)
        append_btn = msgBox.addButton("Append (Skip Duplicates)", QMessageBox.ButtonRole.NoRole)
        cancel_btn = msgBox.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msgBox.setDefaultButton(append_btn)
        msgBox.exec()

        clicked_btn = msgBox.clickedButton()

        # --- Process Import ---
        imported_count = 0
        skipped_count = 0
        hotkey_conflicts = 0

        if clicked_btn == replace_btn:
            print("Replacing data with import...")
            # Clear existing data (DB and memory)
            self._clear_all_internal()

            for item_data in imported_raw_items:
                try:
                    item = ClipboardItem.from_dict(item_data)
                    # Basic validation
                    if not isinstance(item.text, str) or not item.text:
                        skipped_count += 1
                        continue

                    # Add to DB
                    new_id = self.db.add_item(item.text, item.timestamp, item.favorite, item.hotkey)
                    if new_id:
                         item.id = new_id
                         self.clipboard_items.append(item)
                         if item.favorite and item.hotkey:
                             # Check for conflicts *within the imported set* if needed (or let DB unique handle it)
                             self._add_hotkey_binding(item) # Add binding
                         imported_count += 1
                    else:
                         skipped_count += 1
                         print(f"Failed to add imported item to DB: {item.text[:50]}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing imported item {item_data}: {e}", file=sys.stderr)
                    skipped_count += 1
            self.statusBar().showMessage(f"Import complete: Replaced with {imported_count} items ({skipped_count} skipped).", STATUS_TIMEOUT)

        elif clicked_btn == append_btn:
            print("Appending data from import...")
            existing_texts = {item.text for item in self.clipboard_items}
            existing_hotkeys = set(self.hotkeys.keys())

            for item_data in imported_raw_items:
                try:
                    item = ClipboardItem.from_dict(item_data)
                    if not isinstance(item.text, str) or not item.text:
                         skipped_count += 1
                         continue
                    if item.text in existing_texts:
                        skipped_count += 1
                        continue # Skip duplicates

                    # Handle potential hotkey conflicts
                    final_hotkey = item.hotkey
                    if item.favorite and item.hotkey:
                        if item.hotkey in existing_hotkeys:
                            print(f"Hotkey conflict for imported item '{item.text[:30]}...': '{item.hotkey}' already exists. Clearing hotkey.", file=sys.stderr)
                            final_hotkey = None # Clear conflicting hotkey
                            hotkey_conflicts += 1

                    # Add to DB
                    new_id = self.db.add_item(item.text, item.timestamp, item.favorite, final_hotkey)
                    if new_id:
                        item.id = new_id
                        item.hotkey = final_hotkey # Ensure item object has cleared hotkey if needed
                        self.clipboard_items.append(item) # Add to memory
                        existing_texts.add(item.text) # Update for subsequent checks
                        if item.favorite and item.hotkey:
                            self._add_hotkey_binding(item)
                            existing_hotkeys.add(item.hotkey)
                        imported_count += 1
                    else:
                         skipped_count += 1
                         print(f"Failed to add appended item to DB: {item.text[:50]}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing appended item {item_data}: {e}", file=sys.stderr)
                    skipped_count += 1

            # Sort combined list? Current load sorts by timestamp desc. Re-sort here?
            self.clipboard_items.sort(key=lambda x: x.timestamp, reverse=True)
            self.statusBar().showMessage(f"Import complete: Appended {imported_count} new items ({skipped_count} duplicates/skipped, {hotkey_conflicts} hotkey conflicts).", STATUS_TIMEOUT * 1.5)

        elif clicked_btn == cancel_btn:
            self.statusBar().showMessage("Import cancelled.", STATUS_TIMEOUT)
            return # Do nothing

        # Refresh UI after import
        self.refresh_ui()


    def export_data(self):
        """Exports all current clipboard items to a JSON file."""
        if not self.clipboard_items:
            QMessageBox.information(self, "Export", "There is no data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "clipboard_backup.json", "JSON Files (*.json)"
        )
        if not file_path: return

        try:
            # Ensure filename ends with .json
            if not file_path.lower().endswith('.json'):
                file_path += '.json'

            data_to_export = {
                "export_timestamp": datetime.now().isoformat(),
                "clipboard_items": [item.to_dict() for item in self.clipboard_items]
            }

            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data_to_export, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"Exported {len(self.clipboard_items)} items to {os.path.basename(file_path)}", STATUS_TIMEOUT)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")
            self.statusBar().showMessage("Export failed.", STATUS_TIMEOUT)


    def clear_all_data(self):
        """Clears all items from the database and memory after confirmation."""
        reply = QMessageBox.warning(
            self, "Confirm Clear All",
            "<b>Permanently delete all clipboard history and favorites?</b>\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Yes:
            print("Clearing all data...")
            if self._clear_all_internal():
                self.statusBar().showMessage("All clipboard data cleared.", STATUS_TIMEOUT)
            else:
                 self.statusBar().showMessage("Error clearing data from database.", STATUS_TIMEOUT)


    def _clear_all_internal(self):
        """Internal helper to clear DB, memory, and hotkeys."""
        # Clear database
        if not self.db.clear_all():
             return False # DB error

        # Clear hotkeys
        for hotkey_text in list(self.hotkeys.keys()): # Iterate over a copy of keys
            self._remove_hotkey_binding(hotkey_text)
        self.hotkeys = {}

        # Clear memory
        self.clipboard_items = []
        self._last_copied_text = None

        # Refresh UI
        self.refresh_ui()
        return True

    # --- Utility Methods ---
    def _find_item_by_id(self, item_id) -> ClipboardItem | None:
        """Finds an item in the in-memory list by its ID."""
        return next((item for item in self.clipboard_items if item.id == item_id), None)

    def _find_item_by_hotkey(self, hotkey_text) -> ClipboardItem | None:
        """Finds an item in the in-memory list by its hotkey."""
        if not hotkey_text: return None
        return next((item for item in self.clipboard_items if item.hotkey == hotkey_text), None)


    # --- Event Overrides ---
    def closeEvent(self, event):
        """Called when the window is closing."""
        # Stop timers
        self.monitor_timer.stop()
        # No explicit save needed due to SQLite auto-commit behavior
        print("Closing application.")
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon.fromTheme("edit-paste")) # Use a standard icon if available

    # Apply a style if desired (Fusion often looks good cross-platform)
    app.setStyle("Fusion")

    window = ClipboardManager()
    window.show()

    sys.exit(app.exec())

# --- END OF FILE copypaste.py ---