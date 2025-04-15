# wifi_scanner.py
import platform
import subprocess
import re
import sys
from PyQt6.QtCore import QThread, pyqtSignal
from utils import (
    is_valid_bssid,
    parse_signal_strength,
    parse_channel_and_band,
    parse_security,
    parse_standard
)

class WifiScanner(QThread):
    """
    Scans for Wi-Fi networks using platform-specific commands
    and emits the results via signals.
    """
    results_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    @staticmethod
    def list_windows_adapters():
        """
        Returns a list of Wi-Fi adapter names (interface descriptions) on Windows.
        """
        import subprocess, re
        try:
            output = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], encoding="utf-8")
            # Each interface starts with 'Name' or 'Description'
            adapter_names = re.findall(r"^\s*Name\s*:\s*(.+)$", output, re.MULTILINE)
            if not adapter_names:
                # Try Description as fallback
                adapter_names = re.findall(r"^\s*Description\s*:\s*(.+)$", output, re.MULTILINE)
            return adapter_names
        except Exception:
            return []

    def __init__(self, parent=None, adapter_name=None):
        super().__init__(parent)
        self.os_type = platform.system()
        self.adapter_name = adapter_name

    def run(self):
        """Main thread execution method."""
        self.status_signal.emit("Detecting OS and starting scan...")
        networks = []
        try:
            if self.os_type == "Windows":
                self.status_signal.emit("Scanning using netsh (Windows)...")
                networks = self._scan_windows()
            elif self.os_type == "Darwin": # macOS
                self.status_signal.emit("Scanning using airport (macOS)...")
                networks = self._scan_macos()
            elif self.os_type == "Linux":
                self.status_signal.emit("Scanning using nmcli/iwlist (Linux)...")
                networks = self._scan_linux()
            else:
                raise OSError(f"Unsupported operating system: {self.os_type}")

            self.status_signal.emit(f"Parsing complete. Found {len(networks)} networks.")
            self.results_signal.emit(networks)

        except FileNotFoundError as fnf_error:
            self.error_signal.emit(f"Scan command not found: {fnf_error}. Is the required tool installed and in PATH?")
        except subprocess.CalledProcessError as cpe:
            error_output = cpe.stderr.decode(sys.getdefaultencoding(), errors='ignore').strip() if cpe.stderr else "No stderr output."
            self.error_signal.emit(f"Scan command failed (Exit Code {cpe.returncode}): {error_output}. Check permissions or command syntax.")
        except PermissionError:
             self.error_signal.emit("Permission denied. Try running the application with administrator/sudo privileges.")
        except Exception as e:
            self.error_signal.emit(f"An unexpected error occurred during scan: {str(e)}")
        finally:
            self.finished_signal.emit()

    def _execute_command(self, command_args):
        """Executes a command and returns its decoded output."""
        try:
            # Use startupinfo on Windows to hide the console window
            startupinfo = None
            if self.os_type == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(
                command_args,
                capture_output=True,
                text=True,
                check=True, # Raise CalledProcessError on non-zero exit codes
                encoding=sys.getdefaultencoding(), # Try system default encoding first
                errors='ignore', # Ignore decoding errors
                startupinfo=startupinfo
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            # Provide more context for CalledProcessError
            print(f"Error running command: {' '.join(command_args)}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise # Re-raise the exception to be caught in run()
        except FileNotFoundError:
             # Re-raise to be caught in run() with a specific message
             raise FileNotFoundError(f"Command '{command_args[0]}' not found.")
        except PermissionError:
             raise PermissionError("Permission denied executing command.")

    def _scan_windows(self):
        """Scans using 'netsh wlan show networks mode=bssid' on Windows, optionally for a specific adapter."""
        # If adapter_name is specified, use 'interface="name"'
        if self.adapter_name:
            output = self._execute_command([
                "netsh", "wlan", "show", "networks", "mode=bssid", f"interface={self.adapter_name}"])
        else:
            output = self._execute_command(["netsh", "wlan", "show", "networks", "mode=bssid"])
        networks = []
        current_network = {}
        ssid_block_re = re.compile(r"SSID \d+ : (.+?)\n", re.DOTALL)
        network_details_re = re.compile(
            r"Network type\s+:\s*(.+?)\n"
            r"\s*Authentication\s+:\s*(.+?)\n"
            r"\s*Encryption\s+:\s*(.+?)\n"
            r"\s*BSSID 1\s+:\s*([0-9a-fA-F:]+)\s*\n" # Start with BSSID 1
            r"(.*?)", # Capture the rest for signal/channel/etc.
            re.IGNORECASE | re.DOTALL
        )
        bssid_block_re = re.compile(
            r"BSSID \d+\s+:\s*([0-9a-fA-F:]+)\s*\n"
            r"\s*Signal\s+:\s*(\d+%)\s*\n"
            r"\s*Radio type\s+:\s*(.+?)\s*\n" # e.g., 802.11n
            r"\s*Channel\s+:\s*(\d+)\s*\n",
            re.IGNORECASE
        )

        ssid_blocks = ssid_block_re.split(output)[1:] # Skip the initial part before the first SSID

        for i in range(0, len(ssid_blocks), 2):
            ssid_name = ssid_blocks[i].strip()
            ssid_data = ssid_blocks[i+1]

            # Try to parse the main details for this SSID block
            main_details_match = network_details_re.search(ssid_data)
            if not main_details_match: continue # Skip if basic structure isn't found

            _net_type, auth, encrypt, first_bssid, remaining_data = main_details_match.groups()
            security_raw = f"{auth} / {encrypt}".strip()
            security = parse_security(security_raw)

            # Combine the first BSSID block data with the remaining data
            full_bssid_data = f"BSSID 1           : {first_bssid}\n{remaining_data}"

            # Find all BSSIDs associated with this SSID
            bssid_matches = bssid_block_re.findall(full_bssid_data)
            if not bssid_matches: # Sometimes details only appear under BSSID 1 directly
                # Try a simpler regex for the first BSSID if the main one failed
                 simple_bssid_match = re.search(
                     r"Signal\s+:\s*(\d+%)\s*\n"
                     r"\s*Radio type\s+:\s*(.+?)\s*\n"
                     r"\s*Channel\s+:\s*(\d+)\s*\n",
                     remaining_data, re.IGNORECASE)
                 if simple_bssid_match:
                     bssid_matches = [(first_bssid, simple_bssid_match.group(1), simple_bssid_match.group(2), simple_bssid_match.group(3))]


            for bssid, signal_percent, radio_type, channel_str in bssid_matches:
                if not is_valid_bssid(bssid): continue

                signal_dbm = parse_signal_strength(signal_percent)
                channel, band = parse_channel_and_band(channel_str)
                standard = parse_standard(radio_type, channel=channel, band=band)

                networks.append({
                    "ssid": ssid_name,
                    "bssid": bssid.upper(),
                    "signal": signal_dbm, # Store as dBm
                    "channel": channel,
                    "band": band,
                    "standard": standard,
                    "security": security,
                    "manufacturer": "N/A" # Placeholder
                })

        return networks


    def _scan_macos(self):
        """Scans using 'airport -s' on macOS."""
        # Ensure airport is available
        airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
        try:
             # Test execution first without capturing, just to check existence/permissions
             subprocess.run([airport_path, "-I"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
             raise FileNotFoundError(f"{airport_path}. System tool not found or failed. {e}")

        output = self._execute_command([airport_path, "-s"])
        networks = []
        # Regex captures SSID, BSSID, RSSI, Channel, HT/VHT/..., Security
        # Note: SSID can have spaces, even leading/trailing, so we capture greedily before BSSID
        # Example line: "  My Wi-Fi Network       6c:71:d9:xx:xx:xx -55  11      Y    US WPA2(PSK/AES/AES)"
        # Example line: "       My WiFi 5G        f8:e0:79:xx:xx:xx -60  149     Y,-   US WPA2(PSK/AES/AES)" VHT Capable with MCS Index
        # Example line: "       My WiFi 6E        a0:b1:c2:xx:xx:xx -45  5       Y     US WPA3 Per(SAE/AES)" HE Capable
        line_re = re.compile(
            r"^\s*(.+?)\s+([0-9a-fA-F:]+)\s+(-?\d+)\s+(\d+),?(-?\d+)*\s+([YN-])\s+([YN-])\s+([YN-])\s+(\S+)\s+(.*)$", re.IGNORECASE
        )
        # Simpler fallback if the above fails (doesn't parse HT/VHT/HE as well)
        line_re_simple = re.compile(
            r"^\s*(.+?)\s+([0-9a-fA-F:]+)\s+(-?\d+)\s+(\d+)\s+.*?\s+(.*)$"
        )

        lines = output.strip().split('\n')
        if not lines or "SSID" not in lines[0]: # Basic check for header
            return []

        for line in lines[1:]: # Skip header row
            match = line_re.match(line)
            if match:
                ssid, bssid, rssi, channel_str, _ext_chan, ht, vht, he, _country, security_raw = match.groups()
                security_parts = security_raw.split(' ')[0] # Take the first part like 'WPA2(PSK/AES/AES)' or 'WEP' or '--' for Open
            else:
                # Try simpler regex if complex one failed
                match_simple = line_re_simple.match(line)
                if match_simple:
                    ssid, bssid, rssi, channel_str, security_raw = match_simple.groups()
                    security_parts = security_raw.split(' ')[0]
                    ht, vht, he = 'N', 'N', 'N' # Assume no if not parsed
                else:
                    # print(f"Could not parse macOS line: {line}") # Debugging
                    continue # Skip line if parsing fails

            if not is_valid_bssid(bssid): continue

            ssid = ssid.strip()
            signal_dbm = parse_signal_strength(rssi) # Already in dBm (RSSI)
            channel, band = parse_channel_and_band(channel_str)
            security = parse_security(security_parts)

            # Infer standard based on HT/VHT/HE flags and band
            std_list = []
            if band == "6 GHz" or he == 'Y': std_list.append("ax (Wi-Fi 6/6E)")
            if band == "5 GHz" and vht == 'Y': std_list.append("ac (Wi-Fi 5)")
            if (band == "2.4 GHz" or band == "5 GHz") and ht == 'Y': std_list.append("n (Wi-Fi 4)")
            # Basic standards based on channel/band if no flags set or flags not parsed
            if not std_list:
                if band == "5 GHz": std_list.append("a")
                if band == "2.4 GHz": std_list.append("b/g")


            standard = "/".join(std_list) if std_list else parse_standard(None, channel=channel, band=band)


            networks.append({
                "ssid": ssid,
                "bssid": bssid.upper(),
                "signal": signal_dbm,
                "channel": channel,
                "band": band,
                "standard": standard,
                "security": security,
                "manufacturer": "N/A" # Placeholder
            })

        return networks


    def _scan_linux(self):
        """Scans using 'nmcli dev wifi list ifname <iface> --rescan yes' or potentially 'iwlist scan' on Linux."""
        networks = []
        try:
            # Prefer nmcli as it often gives richer info and handles rescanning well
            # Need to find the active Wi-Fi interface first (e.g., wlan0, wlp3s0)
            interface = self._find_linux_wifi_interface_nmcli() or self._find_linux_wifi_interface_iw()
            if not interface:
                 # Try a common default if detection fails
                 interface = "wlan0"
                 self.status_signal.emit(f"Could not detect Wi-Fi interface, trying default '{interface}'...")
                 # raise EnvironmentError("Could not automatically detect a Wi-Fi interface. nmcli or iw/ip tools might be needed.")

            self.status_signal.emit(f"Using interface: {interface}. Running nmcli scan...")
            # --rescan auto/yes might require privileges. Try without first.
            try:
                # Force rescan if possible, requires root or specific polkit permissions
                self._execute_command(["nmcli", "dev", "wifi", "rescan", "ifname", interface])
            except (subprocess.CalledProcessError, PermissionError):
                self.status_signal.emit("nmcli rescan failed (permissions?), trying list only.")
                try:
                    # Try rescan without specifying interface (might work)
                    self._execute_command(["nmcli", "dev", "wifi", "rescan"])
                except (subprocess.CalledProcessError, PermissionError):
                     self.status_signal.emit("nmcli rescan failed again. Using potentially stale data.")

            # Get the list, requesting specific fields
            fields = "SSID,BSSID,SIGNAL,FREQ,CHAN,SECURITY,WPA-FLAGS,RSN-FLAGS"
            # Use '--escape no' to prevent nmcli from adding backslashes before spaces/special chars in SSID
            output = self._execute_command(["nmcli", "-t", "-f", fields, "--escape", "no", "dev", "wifi", "list", "ifname", interface])


            lines = output.strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.split(':')
                if len(parts) < 8: continue # Ensure we have enough fields

                ssid, bssid, signal_str, freq_str, chan_str, security_raw, wpa_flags, rsn_flags = parts[:8]

                if not is_valid_bssid(bssid): continue

                signal_dbm = parse_signal_strength(signal_str) # nmcli signal is 0-100
                channel, band = parse_channel_and_band(chan_str, freq_str)
                security = self._parse_linux_security(security_raw, wpa_flags, rsn_flags)
                # nmcli doesn't directly give 802.11 standard, infer from band/flags
                standard = parse_standard(f"{wpa_flags} {rsn_flags}", channel=channel, band=band) # Pass flags as hint


                networks.append({
                    "ssid": ssid if ssid != "--" else "(Hidden Network)", # nmcli shows '--' for hidden
                    "bssid": bssid.upper(),
                    "signal": signal_dbm,
                    "channel": channel,
                    "band": band,
                    "standard": standard,
                    "security": security,
                    "manufacturer": "N/A" # Placeholder
                })
            return networks

        except FileNotFoundError:
            self.status_signal.emit("nmcli not found. Trying iwlist (may require root)...")
            # Fallback to iwlist scan (parsing is more complex)
            # Note: iwlist often requires root/sudo
            # You would need to implement _scan_linux_iwlist similar to the others
            # For brevity, this fallback is not fully implemented here.
            # raise NotImplementedError("iwlist scan parsing not implemented in this example.")
            self.error_signal.emit("nmcli not found and iwlist fallback is not implemented. Cannot scan on Linux.")
            return []
        except EnvironmentError as e: # Catch interface detection error
             self.error_signal.emit(str(e))
             return []

    def _find_linux_wifi_interface_nmcli(self):
        """Finds the first active Wi-Fi interface using nmcli."""
        try:
            output = self._execute_command(["nmcli", "-t", "-f", "DEVICE,TYPE", "device"])
            for line in output.strip().split('\n'):
                if not line.strip(): continue
                parts = line.split(':')
                if len(parts) == 2 and parts[1].lower() == 'wifi':
                    self.status_signal.emit(f"Found Wi-Fi interface via nmcli: {parts[0]}")
                    return parts[0]
            return None
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError):
            self.status_signal.emit("nmcli check failed, trying 'iw' or 'ip'...")
            return None # Indicate nmcli method failed

    def _find_linux_wifi_interface_iw(self):
        """Finds a Wi-Fi interface using 'iw dev' or 'ip link' as fallback."""
        try: # Try 'iw dev' first
            output = self._execute_command(["iw", "dev"])
            match = re.search(r"Interface\s+([a-zA-Z0-9]+)", output, re.IGNORECASE)
            if match:
                iface = match.group(1)
                self.status_signal.emit(f"Found Wi-Fi interface via iw: {iface}")
                return iface
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError):
            pass # Ignore if 'iw' fails

        try: # Try 'ip link' as another fallback
            output = self._execute_command(["ip", "link", "show"])
            # Look for interfaces starting with 'w' (common convention like wlan, wlp)
            match = re.search(r"^\d+:\s+(wl[a-zA-Z0-9]+):", output, re.MULTILINE)
            if match:
                 iface = match.group(1)
                 self.status_signal.emit(f"Found Wi-Fi interface via ip link: {iface}")
                 return iface
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError):
             pass # Ignore if 'ip' fails

        self.status_signal.emit("Could not reliably detect Wi-Fi interface using nmcli, iw, or ip.")
        return None


    def _parse_linux_security(self, security_raw, wpa_flags, rsn_flags):
        """Parses security info from nmcli output fields."""
        sec_parts = set(p for p in security_raw.split(' ') if p) # Like ['WPA2', '802.1X']
        wpa_parts = set(p for p in wpa_flags.split(' ') if p) # Like ['pair_ccmp', 'group_ccmp', 'psk']
        rsn_parts = set(p for p in rsn_flags.split(' ') if p) # Like ['pair_ccmp', 'group_ccmp', 'psk', 'key_mgmt_sae']

        # Check for WPA3 (SAE)
        if 'key_mgmt_sae' in rsn_parts or 'WPA3' in sec_parts:
            if '802.1X' in sec_parts: return "WPA3-Enterprise"
            return "WPA3-Personal (SAE)"

        # Check for WPA2
        if 'WPA2' in sec_parts:
            if '802.1X' in sec_parts: return "WPA2-Enterprise"
            if 'psk' in wpa_parts or 'psk' in rsn_parts: return "WPA2-PSK"
            return "WPA2" # Generic WPA2 if unsure

        # Check for WPA
        if 'WPA' in sec_parts:
            if '802.1X' in sec_parts: return "WPA-Enterprise"
            if 'psk' in wpa_parts or 'psk' in rsn_parts: return "WPA-PSK"
            return "WPA" # Generic WPA

        # Check for WEP
        if 'WEP' in sec_parts:
            return "WEP"

        # If no security flags found, assume Open
        if not sec_parts and not wpa_parts and not rsn_parts:
            return "Open"

        # Fallback if some flags are present but unparsed
        return security_raw if security_raw else "Unknown"