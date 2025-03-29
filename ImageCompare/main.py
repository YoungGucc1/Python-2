#!/usr/bin/env python3
"""
Image Scanner v4.0
Main entry point for the application.
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, Qt

from view.main_window import MainWindow
from model.scanner_model import ScannerModel
from controller.scanner_controller import ScannerController


def main():
    """Main entry point for the application"""
    # Enable high DPI scaling
    # High DPI scaling is enabled by default in PyQt6
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Image Scanner")
    app.setApplicationVersion("4.0")
    
    # Create MVC components
    model = ScannerModel()
    view = MainWindow()
    controller = ScannerController(model, view)
    
    # Show the main window
    view.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()