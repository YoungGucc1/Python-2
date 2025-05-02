import sys
import os
import logging
import threading
from pathlib import Path
import time # For safe stop check

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox,
    QGroupBox, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt, QObject
from PyQt6.QtGui import QTextCursor

# --- Optional: Dark Theme ---
try:
    import qdarkstyle
    DARK_STYLE_AVAILABLE = True
except ImportError:
    DARK_STYLE_AVAILABLE = False
    # Basic fallback dark theme (use black/grey/white)
    DARK_STYLESHEET = """
        QWidget {
            background-color: #222222; /* Dark Grey Background */
            color: #DDDDDD; /* Light Grey Text */
            border: none;
        }
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #333333; /* Medium Grey Input Background */
            color: #DDDDDD;
            border: 1px solid #555555; /* Darker Grey Border */
            border-radius: 4px;
            padding: 5px;
        }
        QTextEdit { /* Specific styling for the log */
             background-color: #111111; /* Near Black for log */
        }
        QPushButton { /* General Button Style (though none will remain) */
            background-color: #444444; /* Medium Grey */
            color: #DDDDDD;
            border: 1px solid #666666;
            border-radius: 4px;
            padding: 5px 10px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QPushButton:pressed {
            background-color: #333333;
        }
        QPushButton:disabled {
            background-color: #303030;
            color: #888888;
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px 5px 5px;
            color: #CCCCCC; /* Lighter grey for title */
        }
        QLabel {
            color: #DDDDDD;
            padding: 2px;
        }
        QProgressBar { /* Style removed as progress bar is removed */
        }
        QProgressBar::chunk {
        }
        QRadioButton { /* Style removed as radio buttons are removed */
        }
        QRadioButton::indicator::unchecked {
        }
        QRadioButton::indicator::checked {
        }
        /* Style QComboBox dropdown list (though none will remain) */
        QComboBox QAbstractItemView {
            background-color: #333333;
            color: #DDDDDD;
            border: 1px solid #555555;
            selection-background-color: #0078d7; /* Keep selection blue for contrast */
        }
    """

# --- Configuration ---
APP_NAME = "Simple GUI App" # Renamed
# APP_ICON_PATH = "path/to/your/icon.png" # Optional icon path

# --- Logging Handler ---
class QtLogHandler(logging.Handler, QObject):
    """Redirects logging records to a PyQt signal."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self) # Initialize QObject base class

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

# --- Main Application Window ---
class YoloTrainerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_styles()
        self.append_output(f"{APP_NAME} Started.") # Add a startup message

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        # Set window icon if path exists
        # if os.path.exists(APP_ICON_PATH):
        #     self.setWindowIcon(QIcon(APP_ICON_PATH))

        main_layout = QVBoxLayout(self)

        # --- Output Console --- Kept
        output_group = QGroupBox("Output Log") # Renamed title slightly
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.output_text.setPlaceholderText("Log messages will appear here...")
        self.output_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group, 1)

        self.setLayout(main_layout)
        self.resize(600, 400) # Adjusted default size

    def apply_styles(self):
        """Applies the selected stylesheet."""
        # Always apply the basic dark stylesheet now
        self.setStyleSheet(DARK_STYLESHEET)

    def append_output(self, text):
        """Append text to the output log, ensuring it's thread-safe."""
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.output_text.insertPlainText(text.rstrip() + '\n') # Ensure newline
        self.output_text.moveCursor(QTextCursor.MoveOperation.End) # Auto-scroll

    def closeEvent(self, event):
        """Ensure thread is stopped cleanly when closing the window."""
        self.append_output("Exiting application...")
        event.accept()

# --- Run the Application ---
if __name__ == '__main__':
    # Needed for some environments when using multiprocessing within the thread (YOLO might use it)
    # multiprocessing.freeze_support() # Uncomment if you face multiprocessing issues on Windows when packaged

    app = QApplication(sys.argv)
    ex = YoloTrainerApp()
    ex.show()
    sys.exit(app.exec())