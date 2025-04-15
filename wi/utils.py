# utils.py
import re
import platform

# Basic BSSID validation (can be expanded)
BSSID_PATTERN = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')

def is_valid_bssid(mac_address):
    """
    Validates if the given string is a valid BSSID (MAC address).
    """
    return bool(re.match(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$", mac_address))

def is_valid_bssid(mac_address):
    """Checks if a string looks like a valid BSSID/MAC address."""
    return BSSID_PATTERN.match(mac_address) is not None

def parse_signal_strength(signal_str):
    """Attempts to parse signal strength (dBm or percentage) into dBm."""
    try:
        # Try direct dBm conversion
        if "dBm" in signal_str:
            return int(re.sub(r'[^-\d]', '', signal_str))
        # Try percentage conversion (approximate)
        elif "%" in signal_str:
            percent = int(re.sub(r'[^\d]', '', signal_str))
            # Simple linear mapping: 100% -> -30 dBm, 0% -> -100 dBm
            # This is a rough estimate! Real mapping is non-linear.
            if percent > 100: percent = 100
            if percent < 0: percent = 0
            # dBm = (percentage / 2) - 100 # Another common rough conversion
            # Using a slightly different curve common in some drivers
            if percent >= 90: return -55
            if percent >= 80: return -60
            if percent >= 70: return -67
            if percent >= 60: return -70
            if percent >= 50: return -75
            if percent >= 30: return -80
            if percent >= 10: return -85
            return -90
        # Assume it might be just a number (could be dBm or RSSI)
        else:
             num = int(signal_str)
             # If positive, assume RSSI, if negative, assume dBm
             # Very rough guess! Needs OS specific context.
             return num if num < 0 else -num
    except ValueError:
        return -100 # Indicate unknown/poor signal

def parse_channel_and_band(channel_str, freq_str=None):
    """Determines channel and band (2.4, 5, 6 GHz)."""
    try:
        channel = int(channel_str)
        if 1 <= channel <= 14:
            return channel, "2.4 GHz"
        elif 36 <= channel <= 177: # Covers common 5GHz channels
            return channel, "5 GHz"
        elif 1 <= channel <= 233 and platform.system() != "Windows": # Wi-Fi 6E channels (Linux/Mac often report these directly)
             return channel, "6 GHz"
        # Sometimes frequency is provided directly (e.g., nmcli)
        elif freq_str:
            freq_mhz = int(re.sub(r'\D', '', freq_str))
            if 2400 <= freq_mhz < 2500: return channel, "2.4 GHz"
            if 5100 <= freq_mhz < 5900: return channel, "5 GHz"
            if 5900 <= freq_mhz < 7200: return channel, "6 GHz"

        return channel, "Unknown" # Channel number might be outside standard ranges
    except (ValueError, TypeError):
        return "N/A", "Unknown"

def parse_security(sec_str):
    """Normalizes security type string."""
    sec_str_lower = sec_str.lower()
    if "wpa3" in sec_str_lower:
        return "WPA3"
    elif "wpa2-psk" in sec_str_lower or "wpa2-personal" in sec_str_lower:
        return "WPA2-PSK"
    elif "wpa2-enterprise" in sec_str_lower or "wpa2" in sec_str_lower: # Catch broader WPA2
        return "WPA2"
    elif "wpa-psk" in sec_str_lower or "wpa-personal" in sec_str_lower:
        return "WPA-PSK"
    elif "wpa-enterprise" in sec_str_lower or "wpa" in sec_str_lower: # Catch broader WPA
        return "WPA"
    elif "wep" in sec_str_lower:
        return "WEP"
    elif "open" in sec_str_lower or "off" in sec_str_lower or not sec_str: # Check if empty string means open
        return "Open"
    else:
        # Return the original if unsure, maybe it's a specific EAP type etc.
        return sec_str.strip() if sec_str else "Unknown"


def parse_standard(std_str, channel=None, band=None):
    """Attempts to infer Wi-Fi standard (best effort)."""
    # This is highly approximate as CLI tools often don't list the *exact* standard mix.
    std_str_lower = std_str.lower() if std_str else ""

    standards = set()
    if "ax" in std_str_lower or (band == "6 GHz"):
        standards.add("ax (Wi-Fi 6/6E)") # Assume AX on 6GHz
    if "ac" in std_str_lower or (band == "5 GHz" and "ax" not in std_str_lower):
        standards.add("ac (Wi-Fi 5)")
    if "n" in std_str_lower or (band == "2.4 GHz" and "ax" not in std_str_lower):
         standards.add("n (Wi-Fi 4)")
    if "g" in std_str_lower:
        standards.add("g")
    if "a" in std_str_lower and band == "5 GHz": # 'a' is 5GHz only
        standards.add("a")
    if "b" in std_str_lower and band == "2.4 GHz": # 'b' is 2.4GHz only
         standards.add("b")

    # Fallbacks based on band if no specific standard mentioned
    if not standards:
        if band == "6 GHz": standards.add("ax (Wi-Fi 6E)")
        elif band == "5 GHz": standards.add("a/n/ac/ax?")
        elif band == "2.4 GHz": standards.add("b/g/n/ax?")
        else: standards.add("Unknown")

    return "/".join(sorted(list(standards), key=lambda s: ('ax' in s, 'ac' in s, 'n' in s, 'g' in s, 'a' in s, 'b' in s), reverse=True))

# Add OUI lookup function here if implemented later
# def get_manufacturer(bssid):
#    pass