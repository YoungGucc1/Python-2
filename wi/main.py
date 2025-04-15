# main.py
import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Apply system theme or specific style if desired
    # app.setStyle('Fusion') # Example: Use Fusion style

    main_win = MainWindow()
    main_win.show()

    sys.exit(app.exec())import re
    
    def is_valid_bssid(bssid):
        """
        Validates if the given string is a valid BSSID (MAC address).
        """
        return bool(re.match(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$", bssid))