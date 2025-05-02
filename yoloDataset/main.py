"""
YOLO Dataset Creator - Main Entry Point

This application provides a graphical user interface for creating labeled datasets
specifically formatted for training YOLO (You Only Look Once) object detection models.
"""

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt

# Use absolute imports instead of relative imports
from ui.main_window import MainWindow
from core.app_logic import AppLogic

def main():
    """Main entry point for the application"""
    # Set application information
    QCoreApplication.setApplicationName("YOLO Dataset Creator")
    QCoreApplication.setOrganizationName("YOLO")
    QCoreApplication.setApplicationVersion("0.1.0")
    
    # Create the application
    app = QApplication(sys.argv)
    
    # Set fusion style for a consistent look across platforms
    app.setStyle("Fusion")
    
    # Create and display the main window
    main_window = MainWindow()
    
    # Create the application logic
    app_logic = AppLogic(main_window)
    
    # Show the main window
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 