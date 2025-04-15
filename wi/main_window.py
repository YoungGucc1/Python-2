# main_window.py
import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QStatusBar,
    QFileDialog, QHeaderView, QMessageBox, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPalette, QColor

from wifi_scanner import WifiScanner

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Wi-Fi Surveyor")
        self.setGeometry(100, 100, 900, 600) # x, y, width, height

        # Data storage
        self.current_network_data = []

        # --- UI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Adapter selection
        self.adapter_label = QLabel("Wi-Fi Adapter:")
        self.adapter_combo = QComboBox()
        adapters = WifiScanner.list_windows_adapters()
        self.adapter_combo.addItems(adapters if adapters else ["(None detected)"])
        self.layout.addWidget(self.adapter_label)
        self.layout.addWidget(self.adapter_combo)

        # Top layout for controls
        self.controls_layout = QHBoxLayout()
        self.scan_button = QPushButton("📡 Scan Networks")
        self.scan_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.export_button = QPushButton("💾 Export to Excel")
        self.export_button.setStyleSheet("padding: 10px; font-size: 14px;")
        self.export_button.setEnabled(False) # Disabled until first scan
        self.total_count_label = QLabel("Networks Found: 0")
        self.total_count_label.setStyleSheet("font-size: 14px; margin-left: 15px;")

        self.controls_layout.addWidget(self.scan_button)
        self.controls_layout.addWidget(self.export_button)
        self.controls_layout.addStretch(1) # Push count label to the right
        self.controls_layout.addWidget(self.total_count_label)
        self.layout.addLayout(self.controls_layout)

        # Network Table
        self.network_table = QTableWidget()
        self.network_table.setColumnCount(8) # SSID, BSSID, Signal, Channel, Band, Standard, Security, Manufacturer
        self.network_table.setHorizontalHeaderLabels([
            "SSID", "BSSID", "Signal (dBm)", "Channel", "Band", "Standard", "Security", "Manufacturer"
        ])
        self.network_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.network_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.network_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.network_table.setSortingEnabled(True)
        self.network_table.verticalHeader().setVisible(False) # Hide row numbers
        header = self.network_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) # SSID
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # BSSID
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Signal
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Channel
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) # Band
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch) # Standard
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents) # Security
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents) # Manufacturer

        # Set fixed width for some columns to prevent them being too wide
        header.setMinimumSectionSize(80)
        header.resizeSection(1, 140) # BSSID fixed width
        header.resizeSection(2, 100) # Signal fixed width
        header.resizeSection(3, 70) # Channel fixed width
        header.resizeSection(4, 70) # Band fixed width
        header.resizeSection(6, 120) # Security fixed width
        header.resizeSection(7, 120) # Manufacturer fixed width


        self.layout.addWidget(self.network_table)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Click 'Scan Networks' to begin.")

        # --- Worker Thread ---
        self.scanner_thread = WifiScanner(self)

        # --- Connections ---
        self.scan_button.clicked.connect(self.start_scan)
        self.export_button.clicked.connect(self.export_to_excel)
        self.scanner_thread.results_signal.connect(self.update_table)
        self.scanner_thread.error_signal.connect(self.handle_scan_error)
        self.scanner_thread.status_signal.connect(self.update_status)
        self.scanner_thread.finished_signal.connect(self.scan_finished) # Re-enable button etc

        # Apply some basic styling (optional)
        self.apply_styles()


    def apply_styles(self):
        # Basic modern-ish style using QPalette and some CSS
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0; /* Light gray background */
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #cccccc;
                gridline-color: #e0e0e0;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #e0e0e0; /* Slightly darker header */
                padding: 4px;
                border: 1px solid #cccccc;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d7; /* Blue */
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #005a9e; /* Darker blue */
            }
            QPushButton:pressed {
                background-color: #003c6a; /* Even darker blue */
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
                color: #e0e0e0;
            }
            QLabel {
                font-size: 13px;
            }
            QStatusBar {
                font-size: 12px;
            }
        """)
        # You can load external QSS files here too
        # Example:
        # try:
        #     import qdarkstyle
        #     self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
        # except ImportError:
        #     print("QDarkStyleSheet not installed, using default styles.")


    @pyqtSlot()
    def start_scan(self):
        if self.scanner_thread.isRunning():
            self.update_status("Scan already in progress...")
            return

        self.scan_button.setEnabled(False)
        self.export_button.setEnabled(False) # Disable export during scan
        # Clear table immediately for visual feedback
        self.network_table.setRowCount(0)
        self.total_count_label.setText("Networks Found: 0")
        self.current_network_data = [] # Clear previous data
        self.update_status("Starting scan...")
        # Pass selected adapter to scanner
        adapter_name = self.adapter_combo.currentText()
        self.scanner_thread.adapter_name = adapter_name
        self.scanner_thread.start()

    @pyqtSlot(list)
    def update_table(self, networks):
        self.current_network_data = networks # Store data
        self.network_table.setSortingEnabled(False) # Disable sorting during update
        self.network_table.setRowCount(len(networks))

        for row, net in enumerate(networks):
            # Create QTableWidgetItems (ensures sorting works correctly, esp for numbers)
            ssid_item = QTableWidgetItem(net.get("ssid", "N/A"))
            bssid_item = QTableWidgetItem(net.get("bssid", "N/A"))

            signal_val = net.get("signal", -100)
            signal_item = QTableWidgetItem()
            signal_item.setData(Qt.ItemDataRole.DisplayRole, f"{signal_val} dBm") # Display text
            signal_item.setData(Qt.ItemDataRole.EditRole, signal_val) # Store numeric value for sorting

            channel_val = net.get("channel", "N/A")
            channel_item = QTableWidgetItem()
            try: # Store channel numerically if possible
                 channel_num = int(channel_val)
                 channel_item.setData(Qt.ItemDataRole.DisplayRole, str(channel_num))
                 channel_item.setData(Qt.ItemDataRole.EditRole, channel_num)
            except (ValueError, TypeError):
                 channel_item.setData(Qt.ItemDataRole.DisplayRole, str(channel_val)) # Store as string if not number
                 channel_item.setData(Qt.ItemDataRole.EditRole, -1) # Low value for sorting

            band_item = QTableWidgetItem(net.get("band", "N/A"))
            standard_item = QTableWidgetItem(net.get("standard", "N/A"))
            security_item = QTableWidgetItem(net.get("security", "N/A"))
            manufacturer_item = QTableWidgetItem(net.get("manufacturer", "N/A")) # Usually N/A unless OUI lookup added

            # Center align signal, channel, band for readability
            signal_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            channel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            band_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.network_table.setItem(row, 0, ssid_item)
            self.network_table.setItem(row, 1, bssid_item)
            self.network_table.setItem(row, 2, signal_item)
            self.network_table.setItem(row, 3, channel_item)
            self.network_table.setItem(row, 4, band_item)
            self.network_table.setItem(row, 5, standard_item)
            self.network_table.setItem(row, 6, security_item)
            self.network_table.setItem(row, 7, manufacturer_item)

        self.network_table.setSortingEnabled(True)
        # Optional: sort by signal strength descending by default after scan
        self.network_table.sortByColumn(2, Qt.SortOrder.DescendingOrder)

        self.total_count_label.setText(f"Networks Found: {len(networks)}")
        self.update_status(f"Scan complete. Found {len(networks)} networks.")
        if networks:
            self.export_button.setEnabled(True)

    @pyqtSlot(str)
    def handle_scan_error(self, error_message):
        self.update_status(f"Scan Error: {error_message}")
        QMessageBox.warning(self, "Scan Error", f"Could not complete Wi-Fi scan:\n\n{error_message}")
        # Ensure buttons are re-enabled even on error
        self.scan_finished()

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_bar.showMessage(message)

    @pyqtSlot()
    def scan_finished(self):
        """Called when the scanner thread finishes, regardless of success/error."""
        if not self.scanner_thread.isRunning(): # Double check it actually finished
            self.scan_button.setEnabled(True)
            # Only enable export if there's data from the *last* scan attempt
            self.export_button.setEnabled(bool(self.current_network_data))
            if not self.status_bar.currentMessage().startswith("Scan Error"):
                 if self.current_network_data:
                     self.update_status(f"Scan complete. Found {len(self.current_network_data)} networks. Ready for next scan or export.")
                 else:
                     # Check if the error signal was emitted previously
                     if not self.status_bar.currentMessage().startswith("Scan Error"):
                        self.update_status("Scan finished, but no networks found or previous error occurred. Ready.")


    @pyqtSlot()
    def export_to_excel(self):
        if not self.current_network_data:
            QMessageBox.information(self, "Export Error", "No network data to export. Please scan first.")
            return

        default_filename = "wifi_survey_export.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Wi-Fi Data",
            default_filename,
            "Excel Files (*.xlsx);;All Files (*)"
        )

        if file_path:
            self.update_status(f"Exporting data to {file_path}...")
            QApplication.processEvents() # Update GUI status before potentially blocking save

            try:
                # Prepare data for DataFrame (match table columns)
                export_data = []
                for net in self.current_network_data:
                     export_data.append({
                        "SSID": net.get("ssid", "N/A"),
                        "BSSID": net.get("bssid", "N/A"),
                        "Signal (dBm)": net.get("signal", "N/A"),
                        "Channel": net.get("channel", "N/A"),
                        "Band": net.get("band", "N/A"),
                        "Standard": net.get("standard", "N/A"),
                        "Security": net.get("security", "N/A"),
                        "Manufacturer": net.get("manufacturer", "N/A"),
                     })

                df = pd.DataFrame(export_data)

                # Specify columns order explicitly for excel output
                columns_order = ["SSID", "BSSID", "Signal (dBm)", "Channel", "Band", "Standard", "Security", "Manufacturer"]
                df = df[columns_order]

                df.to_excel(file_path, index=False, engine='openpyxl')
                self.update_status(f"Export successful: {file_path}")
                QMessageBox.information(self, "Export Successful", f"Network data exported to:\n{file_path}")

            except PermissionError:
                 self.update_status("Export failed: Permission denied.")
                 QMessageBox.critical(self, "Export Error", f"Permission denied writing to file:\n{file_path}\n\nPlease choose a different location or check permissions.")
            except Exception as e:
                self.update_status(f"Export failed: {str(e)}")
                QMessageBox.critical(self, "Export Error", f"An unexpected error occurred during export:\n\n{str(e)}")
        else:
            self.update_status("Export cancelled.")

    def closeEvent(self, event):
        """Ensure thread is stopped cleanly on exit."""
        if self.scanner_thread.isRunning():
            self.scanner_thread.quit() # Request thread termination
            self.scanner_thread.wait(2000) # Wait up to 2 seconds for it to finish
        event.accept()