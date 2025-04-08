# main.py
import sys
import os
import sqlite3
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, # <--- ADD QListWidgetItem HERE
    QLineEdit, QLabel, QPushButton, QSplitter, QToolBar,
    QMessageBox, QAbstractItemView, QFormLayout, QTextEdit, QFileDialog
)
from PyQt6.QtGui import QAction, QIcon, QClipboard # QIcon needs icon files or resource system
from PyQt6.QtCore import Qt, QTimer, QSize # QSize for icons

# Import local modules
import db_manager
import crypto_utils
from ui_styles import DARK_STYLE_SHEET
from widgets import LoginDialog, SetupDialog, AddEditEntryDialog, PasswordGeneratorDialog

# --- Constants ---
CLIPBOARD_CLEAR_TIMEOUT_MS = 30000 # 30 seconds

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyVaultSecure")
        self.setGeometry(200, 200, 900, 600) # x, y, width, height

        self.derived_key = None # Store the key derived from master password
        self.current_entry_id = None
        self.clipboard_timer = QTimer(self)
        self.clipboard_timer.setSingleShot(True)
        self.clipboard_timer.timeout.connect(self.clear_clipboard_action)
        
        # Store the current database path
        self.current_db_path = db_manager.DB_FILE

        self.setup_ui()
        
        # Start with a locked UI but don't prompt for password yet
        self.lock_ui_only()
        
        # Prompt for database selection immediately
        QTimer.singleShot(100, self.prompt_for_database_selection)

    def setup_ui(self):
        # --- Central Widget and Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0) # No margin around the main layout
        main_layout.setSpacing(0) # No spacing between toolbar and splitter

        # --- Toolbar ---
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(20, 20)) # Adjust icon size
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Placeholder icons (replace with actual QIcon('path/to/icon.png'))
        self.add_action = QAction(QIcon(), "Add Entry", self) # Add icon later
        self.add_action.triggered.connect(self.show_add_entry_dialog)
        self.toolbar.addAction(self.add_action)

        self.gen_action = QAction(QIcon(), "Generate Password", self)
        self.gen_action.triggered.connect(self.show_password_generator_dialog)
        self.toolbar.addAction(self.gen_action)
        
        # Add database selection buttons
        self.db_action = QAction(QIcon(), "Select Database", self)
        self.db_action.triggered.connect(self.select_database)
        self.toolbar.addAction(self.db_action)
        
        self.new_db_action = QAction(QIcon(), "New Database", self)
        self.new_db_action.triggered.connect(self.create_new_database)
        self.toolbar.addAction(self.new_db_action)

        self.toolbar.addSeparator()

        self.lock_action = QAction(QIcon(), "Lock Vault", self)
        self.lock_action.triggered.connect(self.lock_vault)
        self.toolbar.addAction(self.lock_action)
        
        # --- Status Bar ---
        self.status_bar = self.statusBar()
        self.db_path_label = QLabel(f"Database: {os.path.basename(self.current_db_path)}")
        self.status_bar.addPermanentWidget(self.db_path_label)

        # --- Splitter (Left and Right Panes) ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Pane ---
        left_pane_widget = QWidget()
        left_layout = QVBoxLayout(left_pane_widget)
        left_layout.setContentsMargins(5, 5, 5, 5) # Padding inside left pane

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search entries...")
        self.search_input.setObjectName("searchLineEdit") # For specific styling
        self.search_input.textChanged.connect(self.filter_entries)
        left_layout.addWidget(self.search_input)

        self.entry_list = QListWidget()
        self.entry_list.setAlternatingRowColors(False) # Let stylesheet handle colors
        self.entry_list.itemSelectionChanged.connect(self.display_selected_entry)
        self.entry_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left_layout.addWidget(self.entry_list)

        self.splitter.addWidget(left_pane_widget)

        # --- Right Pane ---
        right_pane_widget = QWidget()
        right_layout = QVBoxLayout(right_pane_widget)
        right_layout.setContentsMargins(10, 10, 10, 10) # More padding for details

        self.details_form_layout = QFormLayout()
        self.details_form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows) # Better wrapping

        self.title_display = QLineEdit()
        self.title_display.setReadOnly(True)
        self.username_display = QLineEdit()
        self.username_display.setReadOnly(True)
        self.password_display = QLineEdit()
        self.password_display.setReadOnly(True)
        self.password_display.setEchoMode(QLineEdit.EchoMode.Password)
        self.url_display = QLineEdit()
        self.url_display.setReadOnly(True)
        self.notes_display = QTextEdit()
        self.notes_display.setReadOnly(True)
        self.notes_display.setAcceptRichText(False)

        # Detail buttons layout
        details_buttons_layout = QHBoxLayout()
        details_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignLeft) # Align buttons left

        self.copy_user_button = QPushButton("Copy User")
        self.copy_user_button.setObjectName("copyButton")
        self.copy_user_button.clicked.connect(lambda: self.copy_to_clipboard(self.username_display.text()))

        self.copy_pass_button = QPushButton("Copy Pass")
        self.copy_pass_button.setObjectName("copyButton")
        self.copy_pass_button.clicked.connect(lambda: self.copy_to_clipboard(self.password_display.text(), is_password=True))

        self.show_hide_button = QPushButton("Show")
        self.show_hide_button.setObjectName("copyButton") # Reuse style maybe
        self.show_hide_button.setCheckable(True)
        self.show_hide_button.toggled.connect(self.toggle_password_visibility)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.show_edit_entry_dialog)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setStyleSheet("background-color: #a83232;") # Danger color
        self.delete_button.clicked.connect(self.delete_current_entry)

        details_buttons_layout.addWidget(self.copy_user_button)
        details_buttons_layout.addWidget(self.copy_pass_button)
        details_buttons_layout.addWidget(self.show_hide_button)
        details_buttons_layout.addStretch() # Push edit/delete right
        details_buttons_layout.addWidget(self.edit_button)
        details_buttons_layout.addWidget(self.delete_button)

        # Add rows to form layout
        self.details_form_layout.addRow("Title:", self.title_display)
        self.details_form_layout.addRow("Username:", self.username_display)
        self.details_form_layout.addRow("Password:", self.password_display)
        self.details_form_layout.addRow("URL:", self.url_display)
        self.details_form_layout.addRow(details_buttons_layout) # Add button row below URL
        self.details_form_layout.addRow(QLabel("Notes:")) # Label for Notes
        self.details_form_layout.addRow(self.notes_display) # Notes take full row


        right_layout.addLayout(self.details_form_layout)
        self.splitter.addWidget(right_pane_widget)

        # --- Initial Splitter Sizes ---
        self.splitter.setSizes([250, 650]) # Adjust initial size ratio

        # --- Apply Dark Theme ---
        self.setStyleSheet(DARK_STYLE_SHEET)

    # --- Core Logic Methods ---

    def perform_first_time_setup(self):
        """Guides user through setting the master password."""
        dialog = SetupDialog(self)
        if dialog.exec():
            password = dialog.get_password()
            salt = crypto_utils.generate_salt()
            self.derived_key = crypto_utils.derive_key(password, salt)
            encryption_check = crypto_utils.generate_encryption_check(self.derived_key)

            try:
                db_manager.setup_database() # Create tables first
                db_manager.store_metadata('salt', salt)
                db_manager.store_metadata('encryption_check', encryption_check)
                QMessageBox.information(self, "Setup Complete", "Master password set successfully.")
                self.unlock_ui() # Unlock after successful setup
            except Exception as e:
                QMessageBox.critical(self, "Setup Failed", f"Could not initialize the vault: {e}")
                # Clean up potentially partially created DB?
                if os.path.exists(db_manager.DB_FILE):
                    os.remove(db_manager.DB_FILE)
                self.close()
        else:
            # User cancelled setup
            QMessageBox.warning(self, "Setup Cancelled", "Application cannot run without setup.")
            self.close()

    def prompt_for_login(self):
        """Shows the login dialog and verifies the password."""
        dialog = LoginDialog(self)
        while True: # Keep prompting until success or cancel
            if dialog.exec():
                password = dialog.get_password()
                salt = db_manager.get_metadata('salt')
                stored_check = db_manager.get_metadata('encryption_check')

                if not salt or not stored_check:
                     QMessageBox.critical(self, "Login Error", "Vault metadata missing or corrupt.")
                     self.close()
                     return # Exit the loop and function

                self.derived_key = crypto_utils.derive_key(password, salt)

                if crypto_utils.verify_encryption_check(self.derived_key, stored_check):
                    self.unlock_ui()
                    break # Successful login, exit loop
                else:
                    QMessageBox.warning(self, "Login Failed", "Incorrect Master Password.")
                    self.derived_key = None # Clear incorrect key
                    # Loop continues, dialog will reopen if user clicks OK again
            else:
                # User cancelled login
                self.close()
                break # Exit loop

    def lock_vault(self):
        """Clears sensitive data and locks the UI."""
        # Just lock the UI without prompting for password
        self.lock_ui_only()
        
        # If we have a database selected, prompt for login
        if os.path.exists(self.current_db_path):
            # Check if this database needs setup or login
            salt = db_manager.get_metadata('salt')
            if salt:
                # Database is set up, prompt for login
                self.prompt_for_login()
            else:
                # No salt found, this is a new database that needs setup
                reply = QMessageBox.question(
                    self, 
                    "New Database", 
                    "This database needs to be set up with a master password. Would you like to set it up now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.perform_first_time_setup()
                else:
                    # User chose not to set up, keep the vault locked
                    pass

    def unlock_ui(self):
        """Enables the UI elements after successful login/setup."""
        print("Unlocking UI...")
        if not self.derived_key:
            print("Unlock attempt failed: No derived key.")
            self.lock_vault() # Ensure locked state if key is missing
            return

        self.centralWidget().setEnabled(True)
        self.search_input.setEnabled(True)
        self.entry_list.setEnabled(True)
        self.set_details_pane_enabled(True) # Enable based on selection later
        self.toolbar.setEnabled(True)
        self.lock_action.setEnabled(True)
        self.load_entries()
        self.clear_details_pane() # Start with empty details
        self.set_details_pane_enabled(False) # Keep details disabled until selection


    def load_entries(self):
        """Loads entry titles into the left list pane."""
        if not self.derived_key: return
        self.entry_list.clear()
        try:
            entries = db_manager.get_all_entry_ids_titles(self.derived_key)
            for entry_id, title in entries:
                # CORRECTED line:
                item = QListWidgetItem(title) # Create an instance of QListWidgetItem

                item.setData(Qt.ItemDataRole.UserRole, entry_id) # Store ID with item
                self.entry_list.addItem(item) # Add the created item instance
        except Exception as e:
            # Keep the error handling, but update the format string slightly for clarity
            QMessageBox.critical(self, "Load Error", f"Failed to load entries: {e}")
            self.lock_vault() # Lock if loading fails badly

    def filter_entries(self):
        """Filters the list widget based on search input."""
        filter_text = self.search_input.text().lower()
        for i in range(self.entry_list.count()):
            item = self.entry_list.item(i)
            item.setHidden(filter_text not in item.text().lower())

    def display_selected_entry(self):
        """Fetches and displays details of the selected entry."""
        selected_items = self.entry_list.selectedItems()
        if not selected_items or not self.derived_key:
            self.clear_details_pane()
            self.set_details_pane_enabled(False)
            self.current_entry_id = None
            return

        item = selected_items[0]
        self.current_entry_id = item.data(Qt.ItemDataRole.UserRole)

        try:
            details = db_manager.get_entry_details(self.derived_key, self.current_entry_id)
            if details:
                if 'error' in details:
                     QMessageBox.warning(self, "Decryption Error", details['error'])
                # Even if there was a partial error, display what we could get
                self.title_display.setText(details.get('title', '[Error]'))
                self.username_display.setText(details.get('username', ''))
                self.password_display.setText(details.get('password', ''))
                 # Reset password visibility on selection change
                self.show_hide_button.setChecked(False)
                self.password_display.setEchoMode(QLineEdit.EchoMode.Password)
                self.show_hide_button.setText("Show")

                self.url_display.setText(details.get('url', ''))
                self.notes_display.setText(details.get('notes', ''))
                self.set_details_pane_enabled(True)
            else:
                QMessageBox.warning(self, "Error", f"Could not retrieve details for entry ID {self.current_entry_id}.")
                self.clear_details_pane()
                self.set_details_pane_enabled(False)
                self.current_entry_id = None
        except Exception as e:
            QMessageBox.critical(self, "Display Error", f"Failed to display entry: {e}")
            self.clear_details_pane()
            self.set_details_pane_enabled(False)
            self.current_entry_id = None

    def clear_details_pane(self):
        """Clears all fields in the right details pane."""
        self.title_display.clear()
        self.username_display.clear()
        self.password_display.clear()
        self.password_display.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_hide_button.setChecked(False)
        self.show_hide_button.setText("Show")
        self.url_display.clear()
        self.notes_display.clear()

    def set_details_pane_enabled(self, enabled: bool):
        """Enables or disables controls in the right pane."""
        # Keep display fields technically enabled for copy-paste, but control interaction buttons
        # self.title_display.setEnabled(enabled) # ReadOnly anyway
        # self.username_display.setEnabled(enabled)
        # self.password_display.setEnabled(enabled)
        # self.url_display.setEnabled(enabled)
        # self.notes_display.setEnabled(enabled)

        self.copy_user_button.setEnabled(enabled and bool(self.username_display.text()))
        self.copy_pass_button.setEnabled(enabled and bool(self.password_display.text()))
        self.show_hide_button.setEnabled(enabled and bool(self.password_display.text()))
        self.edit_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)

    def toggle_password_visibility(self, checked):
        """Shows or hides the password in the details view."""
        if checked:
            self.password_display.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_hide_button.setText("Hide")
        else:
            self.password_display.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_hide_button.setText("Show")

    def copy_to_clipboard(self, text: str, is_password=False):
        """Copies text to clipboard and sets a timer to clear it."""
        if not text: return
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            print(f"Copied {'password' if is_password else 'text'} to clipboard.")
            # Start or restart the timer
            self.clipboard_timer.start(CLIPBOARD_CLEAR_TIMEOUT_MS)
        except Exception as e:
             QMessageBox.warning(self, "Clipboard Error", f"Could not copy to clipboard: {e}")

    def clear_clipboard_action(self):
        """Clears the clipboard (if it still contains what we put there - basic check)."""
        # This check is imperfect but prevents clearing if user copied something else manually.
        # A more robust solution is harder across platforms.
        try:
            clipboard = QApplication.clipboard()
            # Simple check: if the clipboard still contains *any* text, clear it.
            # Avoid checking the exact text as it was stored in memory briefly.
            if clipboard.text():
                clipboard.clear(mode=QClipboard.Mode.Clipboard)
                print("Clipboard cleared automatically.")
        except Exception as e:
            print(f"Could not automatically clear clipboard: {e}") # Log silently

    def show_add_entry_dialog(self):
        """Shows the dialog to add a new entry."""
        if not self.derived_key: return
        dialog = AddEditEntryDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            try:
                db_manager.add_entry(
                    self.derived_key, data['title'], data['username'],
                    data['password'], data['url'], data['notes']
                )
                self.load_entries() # Refresh list
                # Optionally select the newly added item
            except Exception as e:
                QMessageBox.critical(self, "Add Error", f"Could not save entry: {e}")

    def show_edit_entry_dialog(self):
        """Shows the dialog to edit the currently selected entry."""
        if not self.current_entry_id or not self.derived_key: return

        try:
            # Fetch current details again to ensure freshness
            current_data = db_manager.get_entry_details(self.derived_key, self.current_entry_id)
            if not current_data or 'error' in current_data:
                 QMessageBox.warning(self, "Edit Error", "Could not load current data for editing.")
                 return

            dialog = AddEditEntryDialog(self, entry_data=current_data)
            if dialog.exec():
                new_data = dialog.get_data()
                db_manager.update_entry(
                    self.derived_key, self.current_entry_id, new_data['title'],
                    new_data['username'], new_data['password'], new_data['url'], new_data['notes']
                )
                # Refresh list and redisplay updated details
                current_row = self.entry_list.currentRow() # Remember selection
                self.load_entries()
                self.entry_list.setCurrentRow(current_row) # Try to reselect
                self.display_selected_entry() # Update details pane

        except Exception as e:
            QMessageBox.critical(self, "Edit Error", f"Could not update entry: {e}")


    def delete_current_entry(self):
        """Deletes the currently selected entry after confirmation."""
        if not self.current_entry_id: return

        reply = QMessageBox.question(self, 'Confirm Delete',
                                     f"Are you sure you want to delete the entry '{self.title_display.text()}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                db_manager.delete_entry(self.current_entry_id)
                self.current_entry_id = None
                self.load_entries() # Refresh the list
                self.clear_details_pane() # Clear details
                self.set_details_pane_enabled(False) # Disable details pane
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Could not delete entry: {e}")

    def show_password_generator_dialog(self):
        """Shows the password generator utility dialog."""
        # This generator is standalone, doesn't automatically fill fields here
        dialog = PasswordGeneratorDialog(self)
        dialog.exec() # User interacts with the dialog, copies if needed

    def select_database(self):
        """Opens a file dialog to select a database file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Database File", "", "SQLite Database (*.db *.sqlite)"
        )
        
        if file_path:
            # Check if the file exists and is a valid database
            if os.path.exists(file_path):
                try:
                    # Test if it's a valid SQLite database
                    conn = sqlite3.connect(file_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    # Update the database path
                    self.current_db_path = file_path
                    db_manager.set_database_path(file_path)
                    
                    # Update status bar
                    self.db_path_label.setText(f"Database: {os.path.basename(file_path)}")
                    
                    # Show success message
                    QMessageBox.information(self, "Database Selected", 
                                           f"Successfully selected database: {os.path.basename(file_path)}")
                    
                    # Ask if user wants to unlock the database now
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Unlock Database")
                    msg_box.setText("Would you like to unlock this database now?")
                    
                    # Create custom buttons
                    unlock_button = msg_box.addButton("Unlock", QMessageBox.ButtonRole.YesRole)
                    later_button = msg_box.addButton("Later", QMessageBox.ButtonRole.NoRole)
                    
                    msg_box.exec()
                    
                    if msg_box.clickedButton() == unlock_button:
                        # Check if this database needs setup or login
                        salt = db_manager.get_metadata('salt')
                        if not salt:
                            # No salt found, this is a new database that needs setup
                            setup_box = QMessageBox(self)
                            setup_box.setWindowTitle("New Database")
                            setup_box.setText("This database needs to be set up with a master password. Would you like to set it up now?")
                            
                            # Create custom buttons
                            setup_button = setup_box.addButton("Setup", QMessageBox.ButtonRole.YesRole)
                            skip_button = setup_box.addButton("Skip", QMessageBox.ButtonRole.NoRole)
                            
                            setup_box.exec()
                            
                            if setup_box.clickedButton() == setup_button:
                                self.perform_first_time_setup()
                            else:
                                # User chose not to set up, keep the vault locked
                                self.lock_ui_only()
                        else:
                            # Database is set up, prompt for login
                            self.prompt_for_login()
                    else:
                        # User chose not to unlock, keep the vault locked
                        self.lock_ui_only()
                    
                except sqlite3.Error:
                    QMessageBox.critical(self, "Invalid Database", 
                                        "The selected file is not a valid SQLite database.")
            else:
                QMessageBox.critical(self, "File Not Found", 
                                    "The selected file does not exist.")

    def create_new_database(self):
        """Opens a file dialog to create a new database file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Create New Database", "", "SQLite Database (*.db *.sqlite)"
        )
        
        if file_path:
            # Ensure the file has a .db extension if none provided
            if not file_path.lower().endswith(('.db', '.sqlite')):
                file_path += '.db'
                
            try:
                # Create a new database file
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                
                # Create the necessary tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title_encrypted BLOB NOT NULL,
                        username_encrypted BLOB,
                        password_encrypted BLOB NOT NULL,
                        url_encrypted BLOB,
                        notes_encrypted BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY UNIQUE NOT NULL,
                        value BLOB NOT NULL
                    )
                """)
                
                conn.commit()
                conn.close()
                
                # Update the database path
                self.current_db_path = file_path
                db_manager.set_database_path(file_path)
                
                # Update status bar
                self.db_path_label.setText(f"Database: {os.path.basename(file_path)}")
                
                # Show success message
                QMessageBox.information(self, "Database Created", 
                                       f"Successfully created new database: {os.path.basename(file_path)}")
                
                # Ask if user wants to set up the database now
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Setup Database")
                msg_box.setText("Would you like to set up this database with a master password now?")
                
                # Create custom buttons
                setup_button = msg_box.addButton("Setup", QMessageBox.ButtonRole.YesRole)
                later_button = msg_box.addButton("Later", QMessageBox.ButtonRole.NoRole)
                
                msg_box.exec()
                
                if msg_box.clickedButton() == setup_button:
                    # Prompt for setup since this is a new database
                    self.perform_first_time_setup()
                else:
                    # User chose not to set up, keep the vault locked
                    self.lock_ui_only()
                
            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Creation Error", 
                                    f"Failed to create database: {str(e)}")
                # Reset to default database path if creation failed
                self.current_db_path = db_manager.DB_FILE
                self.db_path_label.setText(f"Database: {os.path.basename(self.current_db_path)}")

    def closeEvent(self, event):
        """Ensure vault is locked and clipboard cleared on close."""
        print("Close event triggered.")
        self.derived_key = None # Ensure key is cleared from memory
        self.clear_clipboard_action() # Attempt to clear clipboard on exit
        event.accept()

    def lock_ui_only(self):
        """Locks the UI without prompting for password."""
        print("Locking UI...")
        self.derived_key = None
        self.current_entry_id = None
        self.entry_list.clear()
        self.clear_details_pane()
        self.search_input.clear()
        self.search_input.setEnabled(False)
        self.entry_list.setEnabled(False)
        self.set_details_pane_enabled(False)
        self.toolbar.setEnabled(False) # Disable toolbar actions too
        self.lock_action.setEnabled(False) # Disable lock when already locked
        self.centralWidget().setEnabled(False) # Disable main area visually
        
    def prompt_for_database_selection(self):
        """Prompts the user to select or create a database."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Database Selection")
        msg_box.setText("Would you like to select an existing database or create a new one?")
        
        # Create custom buttons
        select_button = msg_box.addButton("Select", QMessageBox.ButtonRole.YesRole)
        create_button = msg_box.addButton("Create", QMessageBox.ButtonRole.NoRole)
        
        msg_box.exec()
        
        # Check which button was clicked
        if msg_box.clickedButton() == select_button:
            # User chose to select an existing database
            self.select_database()
        else:
            # User chose to create a new database
            self.create_new_database()


# --- Application Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Apply the stylesheet globally
    # app.setStyleSheet(DARK_STYLE_SHEET) # Already applied in MainWindow init for better control

    window = MainWindow()
    window.show() # Show the window first

    # Let the window handle the initial lock/login/setup flow
    # window.check_initial_setup() # Moved into lock_vault -> QTimer callback

    sys.exit(app.exec())