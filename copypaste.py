import sys
import json
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QListWidget, QListWidgetItem, QPushButton, QTabWidget, 
                            QLabel, QLineEdit, QMessageBox, QFileDialog,
                            QDialog, QFormLayout)
from PyQt6.QtGui import QIcon, QKeySequence, QColor, QFont, QFontMetrics, QShortcut
from PyQt6.QtCore import Qt, QSize, QTimer, QMimeData
import pyperclip

class ClipboardItem:
    def __init__(self, text, timestamp=None, favorite=False):
        self.text = text
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.favorite = favorite
        self.hotkey = None
    
    def to_dict(self):
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "favorite": self.favorite,
            "hotkey": self.hotkey
        }
    
    @classmethod
    def from_dict(cls, data):
        item = cls(data["text"], data["timestamp"], data["favorite"])
        item.hotkey = data.get("hotkey")
        return item

class HotkeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Hotkey")
        self.setFixedSize(300, 150)
        self.setStyleSheet("""
            QDialog {
                background-color: #2C3E50;
                color: #ECF0F1;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border-radius: 4px;
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        self.hotkey_edit = QLineEdit()
        self.hotkey_edit.setPlaceholderText("Press key combination...")
        self.hotkey_edit.setReadOnly(True)
        form_layout.addRow("Hotkey:", self.hotkey_edit)
        
        layout.addLayout(form_layout)
        
        btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_hotkey)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.key_sequence = None
        
    def keyPressEvent(self, event):
        modifiers = []
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("Ctrl")
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            modifiers.append("Alt")
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("Shift")
        
        if modifiers and event.key() not in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift):
            key_text = QKeySequence(event.key()).toString()
            hotkey_text = '+'.join(modifiers + [key_text])
            
            self.hotkey_edit.setText(hotkey_text)
            self.key_sequence = QKeySequence(hotkey_text)
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def get_hotkey(self):
        return self.hotkey_edit.text() if self.hotkey_edit.text() else None
        
    def clear_hotkey(self):
        self.hotkey_edit.clear()
        self.key_sequence = None

class ClipboardItemWidget(QWidget):
    def __init__(self, parent=None, item_index=None, clipboard_manager=None):
        super().__init__(parent)
        self.item_index = item_index
        self.clipboard_manager = clipboard_manager
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Increase margins
        layout.setSpacing(10)  # Increase spacing
        
        # Content layout (for text and timestamp)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)
        
        # Text area
        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setMinimumWidth(300)
        self.text_label.setStyleSheet("font-size: 14px;")
        content_layout.addWidget(self.text_label)
        
        # Add content layout to main layout
        layout.addLayout(content_layout, 1)  # 1 means it can stretch
        
        # Star button
        self.star_btn = QPushButton("★")
        self.star_btn.setToolTip("Add to favorites")
        self.star_btn.setFixedSize(40, 40)  # Larger button
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: #7F8C8D;
                font-size: 24px;
                font-weight: bold;
                border-radius: 20px;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #2980B9;
                color: white;
            }
        """)
        self.star_btn.clicked.connect(self.add_to_favorites)
        layout.addWidget(self.star_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    
    def set_text(self, text):
        self.text_label.setText(text)
    
    def set_favorite(self, is_favorite):
        if is_favorite:
            self.star_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F39C12;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    border-radius: 20px;
                    padding: 3px;
                }
                QPushButton:hover {
                    background-color: #D35400;
                    color: white;
                }
            """)
            self.star_btn.setToolTip("Remove from favorites")
        else:
            self.star_btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495E;
                    color: #7F8C8D;
                    font-size: 24px;
                    font-weight: bold;
                    border-radius: 20px;
                    padding: 3px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                    color: white;
                }
            """)
            self.star_btn.setToolTip("Add to favorites")
    
    def add_to_favorites(self):
        if self.clipboard_manager and self.item_index is not None:
            self.clipboard_manager.toggle_favorite(self.item_index)

class ClipboardManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Clipboard Manager")
        self.setMinimumSize(600, 500)
        self.clipboard_items = []
        self.hotkeys = {}
        
        # Set up theme
        self.setup_theme()
        
        # Initialize clipboard monitoring
        self.clipboard = QApplication.clipboard()
        self.clipboard.dataChanged.connect(self.on_clipboard_change)
        
        # Set up UI
        self.setup_ui()
        
        # Load saved data
        self.load_data()
        
        # Start auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.save_data)
        self.auto_save_timer.start(60000)  # Save every minute
    
    def setup_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QTabWidget {
                background-color: #2C3E50;
            }
            QTabWidget::pane {
                border: 1px solid #7F8C8D;
                background-color: #34495E;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #34495E;
                color: #ECF0F1;
                border-bottom-color: #7F8C8D;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 8px 12px;
                margin-right: 2px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: #3498DB;
            }
            QTabBar::tab:hover {
                background-color: #2980B9;
            }
            QListWidget {
                background-color: #34495E;
                border-radius: 4px;
                border: 1px solid #7F8C8D;
                color: #ECF0F1;
                font-size: 14px;
                padding: 5px;
            }
            QListWidget::item {
                margin: 5px 0px;
                padding: 8px;
                border-radius: 4px;
                background-color: #2C3E50;
                min-height: 50px;  /* Minimum height for items */
            }
            QListWidget::item:selected {
                background-color: #3498DB;
            }
            QListWidget::item:hover {
                background-color: #2980B9;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 14px;
            }
        """)
    
    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.setIconSize(QSize(30, 30))  # Set icon size
        self.history_list.setSpacing(5)  # Add spacing between items
        self.history_list.setUniformItemSizes(False)  # Allow different size items
        self.history_list.itemDoubleClicked.connect(self.copy_to_clipboard)
        
        history_btn_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.history_list.currentItem()))
        
        self.favorite_btn = QPushButton("Add to Favorites")
        self.favorite_btn.setIcon(QIcon.fromTheme("bookmark-new"))
        self.favorite_btn.clicked.connect(self.add_to_favorites)
        
        self.remove_history_btn = QPushButton("Remove")
        self.remove_history_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.remove_history_btn.clicked.connect(self.remove_from_history)
        
        history_btn_layout.addWidget(self.copy_btn)
        history_btn_layout.addWidget(self.favorite_btn)
        history_btn_layout.addWidget(self.remove_history_btn)
        
        history_layout.addWidget(QLabel("Clipboard History"))
        history_layout.addWidget(self.history_list)
        history_layout.addLayout(history_btn_layout)
        history_tab.setLayout(history_layout)
        
        # Favorites tab
        favorites_tab = QWidget()
        favorites_layout = QVBoxLayout()
        
        self.favorites_list = QListWidget()
        self.favorites_list.itemDoubleClicked.connect(self.copy_to_clipboard)
        
        favorites_btn_layout = QHBoxLayout()
        
        self.copy_fav_btn = QPushButton("Copy")
        self.copy_fav_btn.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_fav_btn.clicked.connect(lambda: self.copy_to_clipboard(self.favorites_list.currentItem()))
        
        self.remove_fav_btn = QPushButton("Remove from Favorites")
        self.remove_fav_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.remove_fav_btn.clicked.connect(self.remove_from_favorites)
        
        self.hotkey_btn = QPushButton("Assign Hotkey")
        self.hotkey_btn.setIcon(QIcon.fromTheme("preferences-desktop-keyboard-shortcuts"))
        self.hotkey_btn.clicked.connect(self.assign_hotkey)
        
        favorites_btn_layout.addWidget(self.copy_fav_btn)
        favorites_btn_layout.addWidget(self.remove_fav_btn)
        favorites_btn_layout.addWidget(self.hotkey_btn)
        
        favorites_layout.addWidget(QLabel("Favorites"))
        favorites_layout.addWidget(self.favorites_list)
        favorites_layout.addLayout(favorites_btn_layout)
        favorites_tab.setLayout(favorites_layout)
        
        # Add tabs to widget
        self.tab_widget.addTab(history_tab, "History")
        self.tab_widget.addTab(favorites_tab, "Favorites")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Add import/export buttons
        import_export_layout = QHBoxLayout()
        
        self.import_btn = QPushButton("Import")
        self.import_btn.setIcon(QIcon.fromTheme("document-open"))
        self.import_btn.clicked.connect(self.import_data)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_btn.clicked.connect(self.export_data)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setIcon(QIcon.fromTheme("edit-clear-all"))
        self.clear_all_btn.clicked.connect(self.clear_all_data)
        
        import_export_layout.addWidget(self.import_btn)
        import_export_layout.addWidget(self.export_btn)
        import_export_layout.addWidget(self.clear_all_btn)
        
        main_layout.addLayout(import_export_layout)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def add_to_history(self, text):
        # Check if text is already in clipboard_items
        for item in self.clipboard_items:
            if item.text == text:
                # Move to top if found
                self.clipboard_items.remove(item)
                self.clipboard_items.insert(0, item)
                self.refresh_ui()
                return
        
        # Add new item
        clipboard_item = ClipboardItem(text)
        self.clipboard_items.insert(0, clipboard_item)
        
        # Limit history to 100 items
        if len(self.clipboard_items) > 100:
            # Remove non-favorite items first
            non_favorites = [i for i in self.clipboard_items if not i.favorite]
            if non_favorites:
                self.clipboard_items.remove(non_favorites[-1])
        
        self.refresh_ui()
    
    def on_clipboard_change(self):
        mime_data = self.clipboard.mimeData()
        if mime_data.hasText():
            text = mime_data.text()
            if text and text.strip():  # Only add non-empty text
                self.add_to_history(text)
    
    def copy_to_clipboard(self, item):
        if item:
            index = item.data(Qt.ItemDataRole.UserRole)
            clip_item = self.clipboard_items[index]
            pyperclip.copy(clip_item.text)
            self.statusBar().showMessage("Copied to clipboard", 2000)
    
    def add_to_favorites(self):
        current_item = self.history_list.currentItem()
        if current_item:
            index = current_item.data(Qt.ItemDataRole.UserRole)
            self.clipboard_items[index].favorite = True
            self.refresh_ui()
            self.statusBar().showMessage("Added to favorites", 2000)
            
            # Switch to favorites tab
            self.tab_widget.setCurrentIndex(1)
    
    def remove_from_favorites(self):
        current_item = self.favorites_list.currentItem()
        if current_item:
            index = current_item.data(Qt.ItemDataRole.UserRole)
            self.clipboard_items[index].favorite = False
            
            # Clear any hotkey
            if self.clipboard_items[index].hotkey:
                shortcut_seq = self.clipboard_items[index].hotkey
                if shortcut_seq in self.hotkeys:
                    self.hotkeys[shortcut_seq].setEnabled(False)
                    del self.hotkeys[shortcut_seq]
                self.clipboard_items[index].hotkey = None
            
            self.refresh_ui()
            self.statusBar().showMessage("Removed from favorites", 2000)
    
    def remove_from_history(self):
        current_item = self.history_list.currentItem()
        if current_item:
            index = current_item.data(Qt.ItemDataRole.UserRole)
            
            # Clear any hotkey if it's a favorite
            if self.clipboard_items[index].hotkey:
                shortcut_seq = self.clipboard_items[index].hotkey
                if shortcut_seq in self.hotkeys:
                    self.hotkeys[shortcut_seq].setEnabled(False)
                    del self.hotkeys[shortcut_seq]
            
            # Remove the item
            del self.clipboard_items[index]
            self.refresh_ui()
            self.statusBar().showMessage("Item removed", 2000)
    
    def assign_hotkey(self):
        current_item = self.favorites_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a favorite item first")
            return
        
        index = current_item.data(Qt.ItemDataRole.UserRole)
        clip_item = self.clipboard_items[index]
        
        dialog = HotkeyDialog(self)
        if clip_item.hotkey:
            dialog.hotkey_edit.setText(clip_item.hotkey)
        
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            new_hotkey = dialog.get_hotkey()
            
            # Remove old hotkey if exists
            if clip_item.hotkey and clip_item.hotkey in self.hotkeys:
                self.hotkeys[clip_item.hotkey].setEnabled(False)
                del self.hotkeys[clip_item.hotkey]
            
            if new_hotkey:
                # Check if hotkey is already assigned
                for item in self.clipboard_items:
                    if item != clip_item and item.hotkey == new_hotkey:
                        QMessageBox.warning(self, "Warning", f"Hotkey {new_hotkey} is already assigned")
                        return
                
                clip_item.hotkey = new_hotkey
                
                # Create new shortcut
                shortcut = QShortcut(QKeySequence(new_hotkey), self)
                shortcut.activated.connect(lambda item=clip_item: self.activate_hotkey(item))
                self.hotkeys[new_hotkey] = shortcut
                
                self.statusBar().showMessage(f"Assigned hotkey: {new_hotkey}", 2000)
            else:
                clip_item.hotkey = None
                self.statusBar().showMessage("Hotkey cleared", 2000)
            
            self.refresh_ui()
    
    def activate_hotkey(self, item):
        pyperclip.copy(item.text)
        self.statusBar().showMessage(f"Hotkey activated: Copied to clipboard", 2000)
    
    def refresh_ui(self):
        # Update history list
        self.history_list.clear()
        
        for i, item in enumerate(self.clipboard_items):
            list_item = QListWidgetItem()
            
            # Create custom widget
            widget = ClipboardItemWidget(self.history_list, i, self)
            
            # Format text for display (limit to first 50 chars)
            display_text = item.text[:50] + ("..." if len(item.text) > 50 else "")
            display_text = display_text.replace("\n", " ")
            
            timestamp = f"{item.timestamp}"
            
            # Format text
            prefix = ""
            if item.hotkey:
                prefix += f"[{item.hotkey}] "
            
            widget.set_text(f"{prefix}{display_text}\n{timestamp}")
            widget.set_favorite(item.favorite)
            
            # Set size hint with extra height and width to ensure visibility
            list_item.setSizeHint(QSize(self.history_list.width() - 30, 60))
            
            # Store the original index as user data
            list_item.setData(Qt.ItemDataRole.UserRole, i)
            
            self.history_list.addItem(list_item)
            self.history_list.setItemWidget(list_item, widget)
        
        # Update favorites list
        self.favorites_list.clear()
        
        for i, item in enumerate(self.clipboard_items):
            if item.favorite:
                list_item = QListWidgetItem()
                
                # Format text for display
                display_text = item.text[:50] + ("..." if len(item.text) > 50 else "")
                display_text = display_text.replace("\n", " ")
                
                # Add hotkey info if available
                prefix = ""
                if item.hotkey:
                    prefix = f"[{item.hotkey}] "
                
                list_item.setText(f"{prefix}{display_text}")
                
                # Store the original index as user data
                list_item.setData(Qt.ItemDataRole.UserRole, i)
                
                self.favorites_list.addItem(list_item)
    
    def get_data_file_path(self):
        # Get data directory
        if sys.platform == "win32":
            data_dir = os.path.join(os.getenv("APPDATA"), "ClipboardManager")
        else:
            data_dir = os.path.join(os.path.expanduser("~"), ".clipboardmanager")
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        return os.path.join(data_dir, "clipboard_data.json")
    
    def save_data(self):
        try:
            # Convert clipboard items to dictionaries
            data = {
                "clipboard_items": [item.to_dict() for item in self.clipboard_items]
            }
            
            with open(self.get_data_file_path(), "w") as f:
                json.dump(data, f, indent=2)
                
            self.statusBar().showMessage("Data saved", 2000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save data: {str(e)}")
    
    def load_data(self):
        try:
            if os.path.exists(self.get_data_file_path()):
                with open(self.get_data_file_path(), "r") as f:
                    data = json.load(f)
                
                # Clear existing data
                self.clipboard_items = []
                
                # Load clipboard items
                for item_data in data.get("clipboard_items", []):
                    item = ClipboardItem.from_dict(item_data)
                    self.clipboard_items.append(item)
                    
                    # Set up hotkeys for favorite items
                    if item.favorite and item.hotkey:
                        shortcut = QShortcut(QKeySequence(item.hotkey), self)
                        shortcut.activated.connect(lambda i=item: self.activate_hotkey(i))
                        self.hotkeys[item.hotkey] = shortcut
                
                self.refresh_ui()
                self.statusBar().showMessage("Data loaded", 2000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data: {str(e)}")
    
    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                imported_items = []
                for item_data in data.get("clipboard_items", []):
                    item = ClipboardItem.from_dict(item_data)
                    imported_items.append(item)
                
                # Ask if user wants to replace or append
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Import Options")
                msgBox.setText("How do you want to import data?")
                replace_btn = msgBox.addButton("Replace All", QMessageBox.ButtonRole.AcceptRole)
                append_btn = msgBox.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
                msgBox.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                
                msgBox.exec()
                
                if msgBox.clickedButton() == replace_btn:
                    # Replace all data
                    self.clipboard_items = imported_items
                    
                    # Reset hotkeys
                    for seq, shortcut in self.hotkeys.items():
                        shortcut.setEnabled(False)
                    self.hotkeys = {}
                    
                    # Set up new hotkeys
                    for item in self.clipboard_items:
                        if item.favorite and item.hotkey:
                            shortcut = QShortcut(QKeySequence(item.hotkey), self)
                            shortcut.activated.connect(lambda i=item: self.activate_hotkey(i))
                            self.hotkeys[item.hotkey] = shortcut
                    
                    self.statusBar().showMessage(f"Imported {len(imported_items)} items (replaced)", 2000)
                
                elif msgBox.clickedButton() == append_btn:
                    # Append new items
                    existing_texts = [item.text for item in self.clipboard_items]
                    
                    added_count = 0
                    for item in imported_items:
                        if item.text not in existing_texts:
                            self.clipboard_items.append(item)
                            
                            # Set up hotkey if needed
                            if item.favorite and item.hotkey:
                                # Check if hotkey is not already in use
                                if item.hotkey not in self.hotkeys:
                                    shortcut = QShortcut(QKeySequence(item.hotkey), self)
                                    shortcut.activated.connect(lambda i=item: self.activate_hotkey(i))
                                    self.hotkeys[item.hotkey] = shortcut
                                else:
                                    # Hotkey conflict - clear it
                                    item.hotkey = None
                            
                            added_count += 1
                            existing_texts.append(item.text)
                    
                    self.statusBar().showMessage(f"Imported {added_count} new items", 2000)
                
                self.refresh_ui()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to import data: {str(e)}")
    
    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                data = {
                    "clipboard_items": [item.to_dict() for item in self.clipboard_items]
                }
                
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
                
                self.statusBar().showMessage(f"Exported {len(self.clipboard_items)} items", 2000)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export data: {str(e)}")
    
    def clear_all_data(self):
        reply = QMessageBox.question(
            self, "Confirm Clear", "Are you sure you want to clear all clipboard data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear all data
            self.clipboard_items = []
            
            # Clear all hotkeys
            for shortcut in self.hotkeys.values():
                shortcut.setEnabled(False)
            self.hotkeys = {}
            
            self.refresh_ui()
            self.statusBar().showMessage("All data cleared", 2000)
    
    def closeEvent(self, event):
        self.save_data()  # Save on exit
        event.accept()

    def toggle_favorite(self, index):
        # Toggle favorite status
        item = self.clipboard_items[index]
        item.favorite = not item.favorite
        
        self.refresh_ui()
        
        if item.favorite:
            self.statusBar().showMessage("Added to favorites", 2000)
        else:
            # Clear any hotkey if removing from favorites
            if item.hotkey:
                shortcut_seq = item.hotkey
                if shortcut_seq in self.hotkeys:
                    self.hotkeys[shortcut_seq].setEnabled(False)
                    del self.hotkeys[shortcut_seq]
                item.hotkey = None
            
            self.statusBar().showMessage("Removed from favorites", 2000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = ClipboardManager()
    window.show()
    
    sys.exit(app.exec())