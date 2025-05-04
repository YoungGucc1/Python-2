# main_window.py

import sys
import pyautogui
import os
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QRadioButton, QLabel, QListWidget, QSpinBox,
    QMessageBox, QStatusBar, QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QThread
from PyQt6.QtGui import QAction, QShortcut, QKeySequence # For menu bar and shortcuts

# Optional: Import styles if using styles.py
try:
    from styles import DARK_STYLE
except ImportError:
    DARK_STYLE = None # Fallback if styles.py doesn't exist

# Import the worker thread
from clicker_thread import ClickerThread

import threading
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False


class MainWindow(QMainWindow):
    """Main application window."""

    MODE_CAPTURE = 0
    MODE_CLICK = 1

    def _apply_styles(self):
        """Apply dark style if available."""
        if DARK_STYLE is not None:
            self.setStyleSheet(DARK_STYLE)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state_file = os.path.join(os.path.dirname(__file__), 'app_state.json')
        self.coordinates = []
        self.current_mode = self.MODE_CAPTURE
        self.is_capturing = False
        self.clicker_thread = None

        # Timer to update cursor position label continuously during capture mode
        self.cursor_pos_timer = QTimer(self)
        self.cursor_pos_timer.timeout.connect(self._update_cursor_pos_label)
        self.cursor_pos_timer.setInterval(100) # Update every 100ms

        self._setup_ui()
        self._apply_styles() # Apply the dark theme
        self.load_app_state()
        # Only call after all buttons are created
        self._update_ui_for_mode()
        # Start global hotkey listener if keyboard is available
        if KEYBOARD_AVAILABLE:
            self._global_hotkey_thread = threading.Thread(target=self._start_global_hotkeys, daemon=True)
            self._global_hotkey_thread.start()
        else:
            print("[WARNING] 'keyboard' module not found. Global hotkeys disabled.")

    def save_app_state(self):
        """Save coordinates, mode, delays, speed, always-on-top to JSON."""
        state = {
            'coordinates': self.coordinates,
            'mode': self.current_mode,
            'min_delay': self.spin_min_delay.value() if hasattr(self, 'spin_min_delay') else 500,
            'max_delay': self.spin_max_delay.value() if hasattr(self, 'spin_max_delay') else 2000,
            'speed_factor': self.spin_speed_factor.value() if hasattr(self, 'spin_speed_factor') else 1,
            'always_on_top': self.check_always_on_top.isChecked() if hasattr(self, 'check_always_on_top') else False
        }
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save app state: {e}")

    def load_app_state(self):
        """Load state from JSON and apply to UI."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Coordinates
            self.coordinates = state.get('coordinates', [])
            self.list_coords.clear()
            for coord in self.coordinates:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    self.list_coords.addItem(f"X: {coord[0]}, Y: {coord[1]}")
            # Mode
            mode = state.get('mode', self.MODE_CAPTURE)
            if mode == self.MODE_CLICK:
                self.radio_click.setChecked(True)
                self.current_mode = self.MODE_CLICK
            else:
                self.radio_capture.setChecked(True)
                self.current_mode = self.MODE_CAPTURE
            # Delays
            self.spin_min_delay.setValue(state.get('min_delay', 500))
            self.spin_max_delay.setValue(state.get('max_delay', 2000))
            self.spin_speed_factor.setValue(state.get('speed_factor', 1))
            # Always on top
            self.check_always_on_top.setChecked(state.get('always_on_top', False))
        except Exception as e:
            print(f"[ERROR] Failed to load app state: {e}")



    def _setup_ui(self):
        """Creates and arranges widgets."""
        self.setWindowTitle("PyAutoClicker")
        self.setGeometry(100, 100, 650, 450) # x, y, width, height

        # --- Central Widget and Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align controls to top

        # Mode Selection
        mode_label = QLabel("Mode:")
        self.radio_capture = QRadioButton("Capture Mode")
        self.radio_click = QRadioButton("Click Mode")
        self.radio_capture.setChecked(True)
        self.radio_capture.toggled.connect(lambda: self._set_mode(self.MODE_CAPTURE))
        self.radio_click.toggled.connect(lambda: self._set_mode(self.MODE_CLICK))

        mode_hbox = QHBoxLayout()
        mode_hbox.addWidget(mode_label)
        mode_hbox.addWidget(self.radio_capture)
        mode_hbox.addWidget(self.radio_click)
        mode_hbox.addStretch()
        left_layout.addLayout(mode_hbox)

        # Action Buttons
        self.btn_toggle_capture = QPushButton("Start Capture Hotkey (F6)")
        self.btn_toggle_capture.setCheckable(True) # Make it a toggle button
        self.btn_toggle_capture.clicked.connect(self._toggle_capture_hotkey)

        self.btn_capture_current = QPushButton("Capture Current Position")
        self.btn_capture_current.clicked.connect(self._capture_coordinate)

        self.btn_toggle_click = QPushButton("Start Clicking")
        self.btn_toggle_click.clicked.connect(self._toggle_clicking)

        self.btn_pause_click = QPushButton("Pause Clicking (F7)")
        self.btn_pause_click.setEnabled(False)
        self.btn_pause_click.clicked.connect(self._pause_clicking)

        self.btn_stop_click = QPushButton("Stop Clicking (F8)")
        self.btn_stop_click.setEnabled(False)
        self.btn_stop_click.clicked.connect(self._stop_clicking)

        self.btn_clear_coords = QPushButton("Clear All Coordinates")
        self.btn_clear_coords.clicked.connect(self._clear_coordinates)

        left_layout.addWidget(self.btn_toggle_capture)
        left_layout.addWidget(self.btn_capture_current)
        left_layout.addWidget(self.btn_toggle_click)
        left_layout.addWidget(self.btn_pause_click)
        left_layout.addWidget(self.btn_stop_click)
        left_layout.addWidget(self.btn_clear_coords)

        # Settings
        settings_label = QLabel("Settings:")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Min Delay (ms):"), 0, 0)
        self.spin_min_delay = QSpinBox()
        self.spin_min_delay.setRange(10, 600000)
        self.spin_min_delay.setValue(500)
        settings_layout.addWidget(self.spin_min_delay, 0, 1)

        settings_layout.addWidget(QLabel("Max Delay (ms):"), 1, 0)
        self.spin_max_delay = QSpinBox()
        self.spin_max_delay.setRange(10, 600000)
        self.spin_max_delay.setValue(2000)
        settings_layout.addWidget(self.spin_max_delay, 1, 1)

        settings_layout.addWidget(QLabel("Speed Factor:"), 2, 0)
        self.spin_speed_factor = QSpinBox() # Use QDoubleSpinBox for float if needed
        self.spin_speed_factor.setRange(1, 10) # 1x to 10x slower (adjust logic if needed)
        self.spin_speed_factor.setValue(1) # 1 = normal speed
        self.spin_speed_factor.setToolTip("Movement duration multiplier (1=normal, >1 slower)")
        settings_layout.addWidget(self.spin_speed_factor, 2, 1)


        left_layout.addWidget(settings_label)
        left_layout.addLayout(settings_layout)
        left_layout.addStretch() # Pushes status bar elements down


        # Always on Top Checkbox
        self.check_always_on_top = QCheckBox("Always on Top")
        self.check_always_on_top.toggled.connect(self._toggle_always_on_top)
        left_layout.addWidget(self.check_always_on_top)


        # --- Right Panel (Coordinates List) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        coords_label = QLabel("Captured Coordinates:")
        self.list_coords = QListWidget()
        self.list_coords.itemDoubleClicked.connect(self._remove_coordinate_item) # Double click to remove

        right_layout.addWidget(coords_label)
        right_layout.addWidget(self.list_coords)

        # Cursor Position Label (for capture mode feedback)
        self.lbl_cursor_pos = QLabel("Cursor: (X, Y)")
        self.lbl_cursor_pos.setVisible(True) # Visible in capture mode
        right_layout.addWidget(self.lbl_cursor_pos)


        # --- Add Panels to Main Layout ---
        main_layout.addWidget(left_panel, 1) # Proportion 1
        main_layout.addWidget(right_panel, 1) # Proportion 1


        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self._update_status("Idle.")

        # --- Hotkeys for Pause/Stop in Click Mode ---
        self.shortcut_pause = QShortcut(QKeySequence("Shift+P"), self)
        self.shortcut_pause.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_pause.activated.connect(lambda: self.pause_function())

        self.shortcut_stop = QShortcut(QKeySequence("Shift+T"), self)
        self.shortcut_stop.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_stop.activated.connect(self._stop_clicking)

        self.shortcut_capture = QShortcut(QKeySequence("Shift+C"), self)
        self.shortcut_capture.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_capture.activated.connect(self._capture_coordinate)


    def _set_mode(self, mode):
        """Sets the current mode and updates the UI."""
        if self.current_mode == mode: return

        # Prevent mode change during clicking
        if self.clicker_thread and self.clicker_thread.isRunning():
            QMessageBox.warning(self, "Mode Change Disabled",
                                "Cannot change mode while clicking is active.")
            # Revert radio button selection
            if mode == self.MODE_CAPTURE:
                self.radio_click.setChecked(True)
            else:
                self.radio_capture.setChecked(True)
            return

        self.current_mode = mode
        self._update_ui_for_mode()
        self._update_status(f"Mode changed to {'Capture' if mode == self.MODE_CAPTURE else 'Click'}.")


    def _update_ui_for_mode(self):
        """Enables/disables UI elements based on the current mode."""
        is_capture_mode = self.current_mode == self.MODE_CAPTURE
        is_clicking_active = bool(self.clicker_thread and self.clicker_thread.isRunning())
        # Defensive: Only setEnabled if buttons exist
        if hasattr(self, 'btn_toggle_capture') and self.btn_toggle_capture is not None:
            self.btn_toggle_capture.setEnabled(is_capture_mode and not is_clicking_active)
        if hasattr(self, 'btn_capture_current') and self.btn_capture_current is not None:
            self.btn_capture_current.setEnabled(is_capture_mode and not is_clicking_active)
        if hasattr(self, 'lbl_cursor_pos') and self.lbl_cursor_pos is not None:
            self.lbl_cursor_pos.setVisible(is_capture_mode)
        if is_capture_mode and hasattr(self, 'btn_toggle_capture') and self.btn_toggle_capture.isChecked():
            if not self.cursor_pos_timer.isActive(): self.cursor_pos_timer.start()
        else:
            if self.cursor_pos_timer.isActive(): self.cursor_pos_timer.stop()
        if hasattr(self, 'btn_toggle_click') and self.btn_toggle_click is not None:
            self.btn_toggle_click.setEnabled(not is_capture_mode and bool(self.coordinates))
        if hasattr(self, 'btn_pause_click') and self.btn_pause_click is not None:
            self.btn_pause_click.setEnabled(not is_capture_mode and is_clicking_active)
        if hasattr(self, 'btn_stop_click') and self.btn_stop_click is not None:
            self.btn_stop_click.setEnabled(not is_capture_mode and is_clicking_active)
        if hasattr(self, 'radio_capture') and self.radio_capture is not None:
            self.radio_capture.setEnabled(not is_clicking_active)
        if hasattr(self, 'radio_click') and self.radio_click is not None:
            self.radio_click.setEnabled(not is_clicking_active)
        if hasattr(self, 'btn_clear_coords') and self.btn_clear_coords is not None:
            self.btn_clear_coords.setEnabled(not is_clicking_active)
        if hasattr(self, 'list_coords') and self.list_coords is not None:
            self.list_coords.setEnabled(not is_clicking_active) # Prevent modification during click
        # Settings - disable during click
        if hasattr(self, 'spin_min_delay') and self.spin_min_delay is not None:
            self.spin_min_delay.setEnabled(not is_clicking_active)
        if hasattr(self, 'spin_max_delay') and self.spin_max_delay is not None:
            self.spin_max_delay.setEnabled(not is_clicking_active)
        if hasattr(self, 'spin_speed_factor') and self.spin_speed_factor is not None:
            self.spin_speed_factor.setEnabled(not is_clicking_active)


    def _update_status(self, message):
        """Updates the status bar text."""
        self.statusBar.showMessage(message)
        print(f"Status: {message}") # Also print to console for debugging


    # --- Coordinate Handling ---

    def _toggle_capture_hotkey(self, checked):
        """Handles the 'Start/Stop Capture Hotkey' toggle button."""
        # NOTE: Actual global hotkey (like F6) requires a library like pynput.
        # This implementation just toggles the state visually and starts/stops
        # the cursor position update timer.
        self.is_capturing = checked
        if checked:
            self.btn_toggle_capture.setText("Stop Capture Hotkey (F6)")
            self._update_status("Hotkey capture active (Simulated - Press 'Capture Current Position')")
            if not self.cursor_pos_timer.isActive(): self.cursor_pos_timer.start()
            # TODO: Start pynput listener here if implemented
        else:
            self.btn_toggle_capture.setText("Start Capture Hotkey (F6)")
            self._update_status("Hotkey capture stopped.")
            if self.cursor_pos_timer.isActive(): self.cursor_pos_timer.stop()
            # TODO: Stop pynput listener here if implemented


    def _capture_coordinate(self):
        """Captures the current mouse position and adds it to the list."""
        if self.current_mode != self.MODE_CAPTURE: return
        try:
            x, y = pyautogui.position()
            coord_tuple = (x, y)
            if coord_tuple not in self.coordinates: # Avoid duplicates
                 self.coordinates.append(coord_tuple)
                 self.list_coords.addItem(f"X: {x}, Y: {y}")
                 self._update_status(f"Captured coordinate: ({x}, {y})")
                 self._update_ui_for_mode() # Enable click button if first coord
            else:
                 self._update_status(f"Coordinate ({x}, {y}) already captured.")

        except Exception as e:
            self._update_status(f"Error capturing coordinate: {e}")
            QMessageBox.warning(self, "Capture Error", f"Could not get mouse position: {e}")


    def _clear_coordinates(self):
        """Clears all captured coordinates after confirmation."""
        if not self.coordinates: return

        reply = QMessageBox.question(self, "Confirm Clear",
                                     "Are you sure you want to clear all coordinates?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.coordinates.clear()
            self.list_coords.clear()
            self._update_status("Coordinates cleared.")
            self._update_ui_for_mode() # Disable click button


    def _remove_coordinate_item(self, item):
        """Removes the double-clicked coordinate item."""
        if self.clicker_thread and self.clicker_thread.isRunning(): return # Don't modify during click

        row = self.list_coords.row(item)
        if 0 <= row < len(self.coordinates):
            removed_coord = self.coordinates.pop(row)
            self.list_coords.takeItem(row) # Remove from list widget
            self._update_status(f"Removed coordinate: {removed_coord}")
            self._update_ui_for_mode()


    def _update_cursor_pos_label(self):
        """Updates the label showing the current cursor position."""
        if self.current_mode == self.MODE_CAPTURE:
             try:
                 x, y = pyautogui.position()
                 self.lbl_cursor_pos.setText(f"Cursor: ({x}, {y})")
             except Exception:
                 self.lbl_cursor_pos.setText("Cursor: (Error)")


    # --- Clicking Logic ---

    def _toggle_clicking(self):
        """Starts or stops the clicking thread."""
        if self.clicker_thread and self.clicker_thread.isRunning():
            # Stop Clicking
            self.clicker_thread.stop()
            self.btn_toggle_click.setText("Stopping...")
            self.btn_toggle_click.setEnabled(False) # Disable until thread confirms finish
            self.btn_pause_click.setEnabled(False)
            self.btn_stop_click.setEnabled(False)
        else:
            # Start Clicking
            if not self.coordinates:
                QMessageBox.warning(self, "No Coordinates", "Please capture coordinates first.")
                return
            if self.current_mode != self.MODE_CLICK:
                QMessageBox.warning(self, "Wrong Mode", "Switch to Click Mode to start clicking.")
                return

            # Validate delays
            min_delay = self.spin_min_delay.value()
            max_delay = self.spin_max_delay.value()
            if min_delay > max_delay:
                QMessageBox.warning(self, "Invalid Settings", "Minimum delay cannot be greater than maximum delay.")
                return

            settings = {
                'min_delay': min_delay,
                'max_delay': max_delay,
                'speed_factor': self.spin_speed_factor.value()
            }

            try:
                self.clicker_thread = ClickerThread(list(self.coordinates), settings) # Pass a copy
                # Connect signals
                self.clicker_thread.status_update.connect(self._update_status)
                self.clicker_thread.finished.connect(self._handle_click_finished)
                self.clicker_thread.start()

                self.btn_toggle_click.setText("Stop Clicking")
                self.btn_pause_click.setEnabled(True)
                self.btn_stop_click.setEnabled(True)
                self._update_ui_for_mode() # Disable controls
            except ValueError as ve: # Catch error if coordinates list is empty (shouldn't happen here due to check)
                 QMessageBox.critical(self, "Error Starting Clicker", str(ve))
                 self.clicker_thread = None
            except Exception as e:
                QMessageBox.critical(self, "Error Starting Clicker", f"An unexpected error occurred: {e}")
                self.clicker_thread = None


    def _pause_clicking(self):
        """Pauses or resumes the clicking thread."""
        if self.clicker_thread and self.clicker_thread.isRunning():
            if hasattr(self.clicker_thread, 'is_paused'):
                if self.clicker_thread.is_paused:
                    self.clicker_thread.resume()
                    self.btn_pause_click.setText("Pause Clicking (F7)")
                    self._update_status("Clicking resumed.")
                else:
                    self.clicker_thread.pause()
                    self.btn_pause_click.setText("Resume Clicking (F7)")
                    self._update_status("Clicking paused.")


    def _stop_clicking(self):
        """Stops the clicking thread."""
        if self.clicker_thread and self.clicker_thread.isRunning():
            self.clicker_thread.stop()
            self.btn_stop_click.setEnabled(False)
            self.btn_pause_click.setEnabled(False)
            self.btn_toggle_click.setEnabled(False)
            self._update_status("Stopping clicking...")


    def keyPressEvent(self, event):
        # No longer needed for F7/F8, handled by QShortcut
        super().keyPressEvent(event)


    def _handle_click_finished(self):
        """Called when the clicker thread signals it has finished."""
        self._update_status("Clicking stopped.")
        self.clicker_thread = None # Release the thread object
        self.btn_toggle_click.setText("Start Clicking")
        self.btn_pause_click.setEnabled(False)
        self.btn_pause_click.setText("Pause Clicking (F7)")
        self.btn_stop_click.setEnabled(False)
        self._update_ui_for_mode() # Re-enable controls


    # --- Other UI ---

    def _toggle_always_on_top(self, checked):
        """Toggles the window's always-on-top status."""
        if checked:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        else:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.show() # Re-show to apply the flag change


    def closeEvent(self, event):
        self.save_app_state()
        # Optionally, clean up keyboard hooks (not strictly needed since daemon thread)
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all() # Clean up hooks
            except Exception as e:
                print(f"[WARNING] Could not unhook keyboard listeners: {e}")
        if self.clicker_thread and self.clicker_thread.isRunning():
            self._stop_clicking() # Ensure thread stops before closing
        event.accept()

    # --- Global Hotkey Handling ---

    def _toggle_capture_hotkey_global(self):
        """Helper function to toggle capture mode via global hotkey."""
        # Simulate button toggle: call the original slot with the opposite of current state
        current_state = self.btn_toggle_capture.isChecked()
        self._toggle_capture_hotkey(not current_state)


    def _start_clicking_from_hotkey(self):
        """Starts the clicking thread, intended for use by global hotkey (F5)."""
        # Only start if not already running
        if self.clicker_thread and self.clicker_thread.isRunning():
            self._update_status("Clicking is already active.")
            return

        # Perform checks before starting
        if not self.coordinates:
            # Can't show QMessageBox from this thread directly
            print("[WARNING] No Coordinates - cannot start clicking.")
            self._update_status("Cannot start: No coordinates captured.")
            return
        if self.current_mode != self.MODE_CLICK:
            print("[WARNING] Wrong Mode - cannot start clicking.")
            self._update_status("Cannot start: Switch to Click Mode first.")
            return

        min_delay = self.spin_min_delay.value()
        max_delay = self.spin_max_delay.value()
        if min_delay > max_delay:
            print("[WARNING] Invalid Settings - cannot start clicking.")
            self._update_status("Cannot start: Min delay > Max delay.")
            return

        settings = {
            'min_delay': min_delay,
            'max_delay': max_delay,
            'speed_factor': self.spin_speed_factor.value()
        }

        try:
            self.clicker_thread = ClickerThread(list(self.coordinates), settings) # Pass a copy
            self.clicker_thread.status_update.connect(self._update_status)
            self.clicker_thread.finished.connect(self._handle_click_finished)
            self.clicker_thread.start()

            # Update UI elements (like button text/state) from main thread
            # This needs to be done carefully, maybe via a custom signal
            # or checking state in _update_ui_for_mode.
            # For now, just update status.
            self._update_status("Clicking started via hotkey.")
            # Schedule a UI update on the main thread
            QTimer.singleShot(0, self._update_ui_for_mode)
            QTimer.singleShot(0, lambda: self.btn_toggle_click.setText("Stop Clicking"))


        except Exception as e:
            print(f"[ERROR] Failed to start clicker via hotkey: {e}")
            self._update_status(f"Error starting clicker: {e}")
            self.clicker_thread = None


    def _start_global_hotkeys(self):
        """Register global hotkeys using the keyboard library.

        This runs in a separate thread. GUI updates must be scheduled
        on the main thread using QTimer.singleShot.
        """
        if not KEYBOARD_AVAILABLE:
            return

        print("[INFO] Starting global hotkey listener...")
        try:
            # Use lambda functions to wrap the QTimer call
            # F5: Start/Toggle Clicking
            keyboard.add_hotkey('f5', lambda: QTimer.singleShot(0, self._toggle_clicking))

            # F6: Toggle Capture Mode
            keyboard.add_hotkey('f6', lambda: QTimer.singleShot(0, self._toggle_capture_hotkey_global))

            # F7: Pause Clicking
            keyboard.add_hotkey('f7', lambda: QTimer.singleShot(0, self._pause_clicking))

            # F8: Stop Clicking
            keyboard.add_hotkey('f8', lambda: QTimer.singleShot(0, self._stop_clicking))

            print("[INFO] Global hotkeys (F5, F6, F7, F8) registered.")
            # Keep the listener thread alive. keyboard.wait() blocks until terminated.
            # Since this runs in a daemon thread, it will exit automatically when the app closes.
            # If the thread wasn't a daemon, keyboard.wait() would prevent app exit
            # unless explicitly stopped (e.g., keyboard.unhook_all() in closeEvent).
            # We'll add unhook_all in closeEvent for good measure.
            keyboard.wait() # Blocks this thread, listening for keys

        except Exception as e:
            print(f"[ERROR] Failed to register global hotkeys: {e}")
            print("         Try running the application with administrator/root privileges.")
        finally:
             # This might be reached if keyboard.wait() is interrupted, though unlikely here.
             print("[INFO] Global hotkey listener stopped.")