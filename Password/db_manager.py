# db_manager.py
import sqlite3
import datetime
from typing import Optional, List, Tuple, Dict, Any
import crypto_utils
import os # Import our crypto functions

# --- Determine the absolute path to the directory containing this script ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Define the database file path relative to the script directory ---
DB_FILE = os.path.join(_SCRIPT_DIR, "vault.db") # <-- Updated line

print(f"Database file location: {DB_FILE}") # Optional: Add print statement for verification

def set_database_path(path: str):
    """Sets the database file path to use for all operations."""
    global DB_FILE
    DB_FILE = path
    print(f"Database path updated to: {DB_FILE}")

def get_connection():
    """Establishes connection to the SQLite database."""
    # This function now uses the correctly calculated DB_FILE path
    return sqlite3.connect(DB_FILE)

def setup_database():
    """Creates necessary tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    # Entries table
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
    # Metadata table for salt, check value, etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY UNIQUE NOT NULL,
            value BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def store_metadata(key: str, value: bytes):
    """Stores or updates a key-value pair in the metadata table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_metadata(key: str) -> Optional[bytes]:
    """Retrieves a value from the metadata table by key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_entry(key: bytes, title: str, username: Optional[str], password: str, url: Optional[str], notes: Optional[str]):
    """Adds a new encrypted entry to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute("""
        INSERT INTO entries (title_encrypted, username_encrypted, password_encrypted, url_encrypted, notes_encrypted, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        crypto_utils.encrypt_data(key, title),
        crypto_utils.encrypt_data(key, username) if username else None,
        crypto_utils.encrypt_data(key, password),
        crypto_utils.encrypt_data(key, url) if url else None,
        crypto_utils.encrypt_data(key, notes) if notes else None,
        now,
        now
    ))
    conn.commit()
    conn.close()

def get_all_entry_ids_titles(key: bytes) -> List[Tuple[int, str]]:
    """Retrieves all entry IDs and their decrypted titles."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title_encrypted FROM entries ORDER BY LOWER(SUBSTR(CAST(title_encrypted AS TEXT), 13))") # Basic sort, needs decryption
    results = []
    encrypted_rows = cursor.fetchall()
    conn.close()

    for entry_id, title_encrypted in encrypted_rows:
        try:
            title = crypto_utils.decrypt_data(key, title_encrypted)
            results.append((entry_id, title))
        except ValueError:
             # Handle case where one entry might be corrupt, maybe log it
            print(f"Warning: Could not decrypt title for entry ID {entry_id}")
            results.append((entry_id, "[Decryption Error]"))

    # Sort alphabetically after decryption
    results.sort(key=lambda item: item[1].lower())
    return results


def get_entry_details(key: bytes, entry_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves and decrypts all details for a specific entry ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title_encrypted, username_encrypted, password_encrypted, url_encrypted, notes_encrypted
        FROM entries WHERE id = ?
    """, (entry_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    details = {}
    try:
        details['id'] = entry_id
        details['title'] = crypto_utils.decrypt_data(key, row[0]) if row[0] else ''
        details['username'] = crypto_utils.decrypt_data(key, row[1]) if row[1] else ''
        details['password'] = crypto_utils.decrypt_data(key, row[2]) if row[2] else ''
        details['url'] = crypto_utils.decrypt_data(key, row[3]) if row[3] else ''
        details['notes'] = crypto_utils.decrypt_data(key, row[4]) if row[4] else ''
        return details
    except ValueError:
        print(f"Error decrypting details for entry ID {entry_id}")
        # Return partially decrypted data or None/Error indicator
        details['error'] = "Decryption failed for one or more fields."
        # Fill with placeholders for safety if needed
        details.setdefault('title', '[Decryption Error]')
        details.setdefault('username', '')
        details.setdefault('password', '***') # Mask if error
        details.setdefault('url', '')
        details.setdefault('notes', '')
        return details # Or return None

def update_entry(key: bytes, entry_id: int, title: str, username: Optional[str], password: str, url: Optional[str], notes: Optional[str]):
    """Updates an existing encrypted entry."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute("""
        UPDATE entries
        SET title_encrypted = ?, username_encrypted = ?, password_encrypted = ?,
            url_encrypted = ?, notes_encrypted = ?, updated_at = ?
        WHERE id = ?
    """, (
        crypto_utils.encrypt_data(key, title),
        crypto_utils.encrypt_data(key, username) if username else None,
        crypto_utils.encrypt_data(key, password),
        crypto_utils.encrypt_data(key, url) if url else None,
        crypto_utils.encrypt_data(key, notes) if notes else None,
        now,
        entry_id
    ))
    conn.commit()
    conn.close()

def delete_entry(entry_id: int):
    """Deletes an entry from the database by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()