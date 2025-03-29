import streamlit as st
import json
import os
import datetime
import pyperclip
import random
import string
import sqlite3
import tkinter as tk
from tkinter import filedialog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Set page title and configuration
st.set_page_config(page_title="Secure Password Manager", layout="wide", initial_sidebar_state="collapsed")

# CSS to improve UI
st.markdown("""
<style>
    /* Modern, comfortable and colorful design */
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling */
    div[data-testid="column"] > div:has(div.element-container) {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="column"] > div:has(div.element-container):hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #4361ee;
        color: white;
    }
    
    /* Input fields */
    div[data-baseweb="input"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #4361ee;
        box-shadow: 0 0 0 2px rgba(67, 97, 238, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #4361ee;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    
    /* Password strength indicators */
    .password-strength-high {
        color: #10b981;
        font-weight: bold;
    }
    
    .password-strength-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    
    .password-strength-low {
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c5c5c5;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Hide default Streamlit header */
    header {
        visibility: hidden;
    }
    
    /* Custom header styling */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Footer styling */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Function to derive encryption key from password
def get_key_from_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

# Function to encrypt data
def encrypt_data(data, password, salt):
    key, _ = get_key_from_password(password, salt)
    f = Fernet(key)
    encrypted_data = f.encrypt(json.dumps(data).encode())
    return encrypted_data

# Function to decrypt data
def decrypt_data(encrypted_data, password, salt):
    key, _ = get_key_from_password(password, salt)
    f = Fernet(key)
    try:
        decrypted_data = f.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    except Exception:
        return None

# Function to save encrypted data
def save_encrypted_data(data, password):
    # Get the database path from session state
    db_path = st.session_state.get('db_path')
    if not db_path:
        st.error("No database selected. Please log out and select a database.")
        return False
    
    # Get salt or create a new one
    salt = None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value BLOB
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS passwords (
        id INTEGER PRIMARY KEY,
        encrypted_data BLOB,
        created_at TEXT,
        last_modified TEXT
    )
    ''')
    
    # Check if salt exists
    cursor.execute("SELECT value FROM metadata WHERE key = 'salt'")
    result = cursor.fetchone()
    
    if result:
        salt = result[0]
    else:
        salt = os.urandom(16)
        cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ('salt', salt))
    
    # Add last modified timestamp
    data_with_metadata = {
        "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passwords": data
    }
    
    encrypted_data = encrypt_data(data_with_metadata, password, salt)
    
    # Check if we need to update or insert
    cursor.execute("SELECT id FROM passwords LIMIT 1")
    result = cursor.fetchone()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if result:
        # Update existing record
        cursor.execute(
            "UPDATE passwords SET encrypted_data = ?, last_modified = ? WHERE id = ?", 
            (encrypted_data, timestamp, result[0])
        )
    else:
        # Insert new record
        cursor.execute(
            "INSERT INTO passwords (encrypted_data, created_at, last_modified) VALUES (?, ?, ?)",
            (encrypted_data, timestamp, timestamp)
        )
    
    # Create backup
    backup_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cursor.execute(
        "INSERT INTO passwords (encrypted_data, created_at, last_modified) VALUES (?, ?, ?)",
        (encrypted_data, timestamp, f"BACKUP_{backup_timestamp}")
    )
    
    # Keep only the 5 most recent backups
    cursor.execute(
        "DELETE FROM passwords WHERE last_modified LIKE 'BACKUP_%' AND id NOT IN ("
        "SELECT id FROM passwords WHERE last_modified LIKE 'BACKUP_%' ORDER BY id DESC LIMIT 5)"
    )
    
    conn.commit()
    conn.close()
    
    return True

# Function to load encrypted data
def load_encrypted_data(password):
    # Get the database path from session state
    db_path = st.session_state.get('db_path')
    if not db_path or not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='metadata' OR name='passwords')")
        tables = cursor.fetchall()
        if len(tables) < 2:
            conn.close()
            return None
        
        # Get salt
        cursor.execute("SELECT value FROM metadata WHERE key = 'salt'")
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
            
        salt = result[0]
        
        # Get latest password data (excluding backups)
        cursor.execute("SELECT encrypted_data FROM passwords WHERE last_modified NOT LIKE 'BACKUP_%' ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
            
        encrypted_data = result[0]
        conn.close()
        
        decrypted_data = decrypt_data(encrypted_data, password, salt)
        
        if decrypted_data is None:
            return None
        
        # Handle both new format (with metadata) and old format
        if isinstance(decrypted_data, dict) and "passwords" in decrypted_data:
            st.session_state['last_modified'] = decrypted_data.get("last_modified", "Unknown")
            return decrypted_data["passwords"]
        else:
            return decrypted_data
            
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None

# Function to generate a secure password
def generate_password(length=16, include_uppercase=True, include_digits=True, include_special=True):
    chars = string.ascii_lowercase
    if include_uppercase:
        chars += string.ascii_uppercase
    if include_digits:
        chars += string.digits
    if include_special:
        chars += string.punctuation
    
    # Ensure at least one character from each selected type
    password = []
    if include_uppercase:
        password.append(random.choice(string.ascii_uppercase))
    if include_digits:
        password.append(random.choice(string.digits))
    if include_special:
        password.append(random.choice(string.punctuation))
    
    # Fill the rest with random characters
    password.extend(random.choice(chars) for _ in range(length - len(password)))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)

# Function to check password strength
def check_password_strength(password):
    score = 0
    
    # Length check
    if len(password) >= 12:
        score += 3
    elif len(password) >= 8:
        score += 2
    elif len(password) >= 6:
        score += 1
    
    # Character variety check
    if any(c.isupper() for c in password):
        score += 1
    if any(c.islower() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(c in string.punctuation for c in password):
        score += 1
    
    # Score interpretation
    if score >= 6:
        return "High", "password-strength-high"
    elif score >= 4:
        return "Medium", "password-strength-medium"
    else:
        return "Low", "password-strength-low"

# Function to copy to clipboard
def copy_to_clipboard(text):
    pyperclip.copy(text)

# Initialize session state for storing app states
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'master_password' not in st.session_state:
    st.session_state['master_password'] = ""
if 'password_data' not in st.session_state:
    st.session_state['password_data'] = {}
if 'first_time_setup' not in st.session_state:
    st.session_state['first_time_setup'] = False
if 'db_path' not in st.session_state:
    st.session_state['db_path'] = ""
if 'db_selected' not in st.session_state:
    st.session_state['db_selected'] = False
if 'create_new_db' not in st.session_state:
    st.session_state['create_new_db'] = False
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
if 'last_modified' not in st.session_state:
    st.session_state['last_modified'] = "Never"
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = "grid"  # 'grid' or 'list'
if 'category_filter' not in st.session_state:
    st.session_state['category_filter'] = "All"
if 'show_password' not in st.session_state:
    st.session_state['show_password'] = {}

# Function to handle authentication
def authenticate():
    if st.session_state['password_input'] == "":
        st.warning("Please enter a password")
        return
    
    # Check if database path is set
    if 'db_path' not in st.session_state or not st.session_state['db_path']:
        st.error("Please select a database vault first")
        return
    
    if st.session_state['first_time_setup']:
        # Check if password confirmation matches
        if st.session_state.get('password_confirm', "") != st.session_state['password_input']:
            st.error("Passwords do not match!")
            return
            
        st.session_state['authenticated'] = True
        st.session_state['master_password'] = st.session_state['password_input']
        st.session_state['password_data'] = {}
        st.success("Master password set successfully!")
    else:
        data = load_encrypted_data(st.session_state['password_input'])
        if data is not None:
            st.session_state['authenticated'] = True
            st.session_state['master_password'] = st.session_state['password_input']
            st.session_state['password_data'] = data
            st.success("Authentication successful!")
        else:
            st.error("Authentication failed. Invalid password or database.")

# Function to select database file
def select_database():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    
    # Get file path
    if st.session_state['create_new_db']:
        db_path = filedialog.asksaveasfilename(
            title="Create New Password Vault",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            defaultextension=".db"
        )
    else:
        db_path = filedialog.askopenfilename(
            title="Select Password Vault",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
    
    if db_path:
        st.session_state['db_path'] = db_path
        st.session_state['db_selected'] = True
        
        # Check if this is a new database
        if st.session_state['create_new_db'] or not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            st.session_state['first_time_setup'] = True
        else:
            # Existing database
            st.session_state['first_time_setup'] = False
    
    root.destroy()

# Function to save passwords
def save_passwords():
    success = save_encrypted_data(st.session_state['password_data'], st.session_state['master_password'])
    if success:
        st.session_state['last_modified'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("Passwords saved successfully!")
    else:
        st.error("Failed to save passwords.")

# Function to log out
def logout():
    st.session_state['authenticated'] = False
    st.session_state['master_password'] = ""
    st.session_state['password_data'] = {}
    st.session_state['db_selected'] = False
    st.session_state['db_path'] = ""
    st.session_state['search_query'] = ""
    st.session_state['category_filter'] = "All"
    st.session_state['view_mode'] = "grid"
    st.session_state['show_password'] = {}
    st.success("Logged out successfully!")
    st.rerun()

# Function to toggle password visibility
def toggle_password_visibility(service):
    if service in st.session_state['show_password']:
        st.session_state['show_password'].pop(service)
    else:
        st.session_state['show_password'][service] = True

# Main app interface
# Replace the title with a modern, colorful header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem; background: linear-gradient(90deg, #4361ee, #3a0ca3); padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 2rem; color: white; margin-right: 1rem;">🔒</div>
    <div>
        <h1 style="margin: 0; color: white; font-weight: 600; font-size: 2rem;">Secure Password Manager</h1>
        <p style="margin: 0; color: rgba(255, 255, 255, 0.8); font-size: 1rem;">Keep your passwords safe and organized</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['authenticated']:
    # Authentication screen
    st.markdown("""
    <div style="background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
        <h2 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Secure Password Manager</h2>
        <p style="color: #64748b; margin-bottom: 1.5rem;">Please select or create a password vault database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Database selection
    if not st.session_state.get('db_selected', False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; height: 100%;">
                <h3 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Open Existing Vault</h3>
                <p style="color: #64748b;">Access your existing password database.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Select Existing Vault", use_container_width=True):
                st.session_state['create_new_db'] = False
                select_database()
        
        with col2:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; height: 100%;">
                <h3 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Create New Vault</h3>
                <p style="color: #64748b;">Start fresh with a new password database.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Create New Vault", use_container_width=True):
                st.session_state['create_new_db'] = True
                select_database()
        
        # Display selected database
        if st.session_state.get('db_path'):
            st.success(f"Selected vault: {st.session_state['db_path']}")
            st.session_state['db_selected'] = True
            st.rerun()
    
    # Password entry after database selection
    elif st.session_state['first_time_setup']:
        # First time setup screen with modern design
        st.markdown("""
        <div style="background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
            <h2 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Set Up Your Vault</h2>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Please set a master password for your new vault.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display selected database
        st.info(f"Vault: {st.session_state['db_path']}")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.text_input("Create Master Password", type="password", key="password_input")
            st.text_input("Confirm Master Password", type="password", key="password_confirm")
            
            # Password strength indicator
            if st.session_state.get('password_input'):
                strength, css_class = check_password_strength(st.session_state['password_input'])
                st.markdown(f"Password Strength: <span class='{css_class}'>{strength}</span>", unsafe_allow_html=True)
                
                if strength == "Low":
                    st.warning("Consider using a stronger password for better security.")
            
            st.button("Create Vault", on_click=authenticate, 
                     use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; height: 100%;">
                <h3 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Password Tips</h3>
                <ul style="color: #64748b; padding-left: 1.5rem;">
                    <li>Use at least 12 characters</li>
                    <li>Include uppercase and lowercase letters</li>
                    <li>Add numbers and special characters</li>
                    <li>Avoid common words or phrases</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        # Login screen with modern design
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div style="background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
                <h2 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Welcome Back</h2>
                <p style="color: #64748b; margin-bottom: 1.5rem;">Enter your master password to access your passwords.</p>
            </div>
            """, unsafe_allow_html=True)
                
            # Display selected database
            st.info(f"Vault: {st.session_state['db_path']}")
            
            # Password input
            st.text_input("Master Password", type="password", key="password_input")
            st.button("Login", on_click=authenticate, use_container_width=True)
            
            # Option to change database
            if st.button("Change Vault", key="change_vault"):
                st.session_state['db_selected'] = False
                st.rerun()
        
        with col2:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; height: 100%;">
                <h3 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">Security Tips</h3>
                <ul style="color: #64748b; padding-left: 1.5rem;">
                    <li>Never share your master password</li>
                    <li>Make sure no one is watching</li>
                    <li>Use a strong, unique password</li>
                    <li>Log out when you're done</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
else:
    # Password management screen
    st.sidebar.markdown("""
    <div style="padding: 1rem 0;">
        <h2 style="color: #1e293b; font-weight: 600; font-size: 1.5rem; margin-bottom: 1rem;">Options</h2>
    </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.sidebar.selectbox("Mode", ["Password Manager", "Settings", "Backup & Restore", "Password Generator"])
    
    # Add logout button to sidebar
    st.sidebar.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.sidebar.button("Logout", on_click=logout, use_container_width=True)
    
    if app_mode == "Password Manager":
        # Modern header for password vault
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <div>
                <h2 style="color: #1e293b; margin: 0; font-weight: 600;">Your Password Vault</h2>
                <p style="color: #64748b; margin: 0.5rem 0 0 0;">Last modified: {st.session_state['last_modified']}</p>
            </div>
            <div style="background-color: #4361ee; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.875rem;">
                {len(st.session_state['password_data'])} Passwords
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter and search options with modern design
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Search & Filter</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.session_state['search_query'] = st.text_input("🔍 Search passwords", value=st.session_state['search_query'], 
                                                           placeholder="Search by service, username, or notes...")
        
        with col2:
            # Extract all categories from the password data
            all_categories = set()
            for service_details in st.session_state['password_data'].values():
                if "category" in service_details:
                    all_categories.add(service_details["category"])
            
            st.session_state['category_filter'] = st.selectbox(
                "Filter by category", 
                ["All"] + sorted(list(all_categories))
            )
        
        with col3:
            st.session_state['view_mode'] = st.radio("View as", ["Grid", "List"], horizontal=True,
                                                    index=0 if st.session_state['view_mode'] == "grid" else 1)
            st.session_state['view_mode'] = st.session_state['view_mode'].lower()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add/Edit Password Form with modern design
        with st.expander("Add/Edit Password", expanded=False):
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.25rem;">Add or Update Password</h3>
                <p style="color: #64748b; margin: 0;">Fill in the details below to add a new password or update an existing one.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_service = st.text_input("Service/Website Name", placeholder="e.g., Google, Facebook, Twitter")
                new_username = st.text_input("Username/Email", placeholder="Your login username or email")
                new_password = st.text_input("Password", type="password")
                new_category = st.text_input("Category (optional)", placeholder="e.g., Social, Work, Finance")
                
                if new_password:
                    strength, css_class = check_password_strength(new_password)
                    st.markdown(f"Password Strength: <span class='{css_class}'>{strength}</span>", unsafe_allow_html=True)
            
            with col2:
                new_notes = st.text_area("Notes (optional)", placeholder="Add any additional information here...")
                new_url = st.text_input("Website URL (optional)", placeholder="https://example.com")
                
                # Generate password option with modern design
                st.markdown("""
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.25rem;">Password Generator</h3>
                    <p style="color: #64748b; margin: 0 0 0.5rem 0; font-size: 0.875rem;">Create a strong, random password</p>
                </div>
                """, unsafe_allow_html=True)
                
                pw_length = st.slider("Length", min_value=8, max_value=32, value=16)
                col1, col2, col3 = st.columns(3)
                with col1:
                    use_uppercase = st.checkbox("Uppercase", value=True)
                with col2:
                    use_digits = st.checkbox("Digits", value=True)
                with col3:
                    use_special = st.checkbox("Special Chars", value=True)
                
                gen_col1, gen_col2 = st.columns([2, 1])
                with gen_col1:
                    if st.button("Generate Password", use_container_width=True):
                        generated_pw = generate_password(
                            length=pw_length,
                            include_uppercase=use_uppercase,
                            include_digits=use_digits,
                            include_special=use_special
                        )
                        # We need to update session state to show the generated password
                        st.session_state['generated_password'] = generated_pw
                
                # Display generated password if available
                if 'generated_password' in st.session_state:
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem; border-left: 4px solid #4361ee;">
                        <p style="font-family: monospace; margin: 0; word-break: break-all; font-size: 1rem;">
                    """, unsafe_allow_html=True)
                    st.code(st.session_state['generated_password'], language=None)
                    
                    with gen_col2:
                        if st.button("Use This", use_container_width=True):
                            # Set the generated password to the password field
                            st.session_state['new_password'] = st.session_state['generated_password']
            
            # Add button to save the new entry with modern design
            st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
            if st.button("Add/Update Entry", use_container_width=True):
                if new_service:
                    entry = {
                        "username": new_username,
                        "password": new_password,
                        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Add optional fields if provided
                    if new_category:
                        entry["category"] = new_category
                    if new_notes:
                        entry["notes"] = new_notes
                    if new_url:
                        entry["url"] = new_url
                        
                    # If updating an existing entry, preserve creation date
                    if new_service in st.session_state['password_data']:
                        entry["created"] = st.session_state['password_data'][new_service].get(
                            "created", entry["created"])
                    
                    st.session_state['password_data'][new_service] = entry
                    st.success(f"Added/Updated entry for {new_service}")
                    
                    # Auto-save
                    save_passwords()
                else:
                    st.warning("Please enter a service name")
        
        # Filter and search the password data
        filtered_data = {}
        for service, details in st.session_state['password_data'].items():
            # Apply category filter
            if (st.session_state['category_filter'] != "All" and 
                details.get("category") != st.session_state['category_filter']):
                continue
                
            # Apply search filter
            search_query = st.session_state['search_query'].lower()
            if (search_query and 
                not (search_query in service.lower() or 
                     search_query in details.get("username", "").lower() or 
                     search_query in details.get("notes", "").lower() or
                     search_query in details.get("category", "").lower())):
                continue
                
            # Add to filtered results
            filtered_data[service] = details
        
        # Display passwords based on view mode
        if not filtered_data:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <h3 style="color: #1e293b; margin-bottom: 0.5rem; font-weight: 600;">No passwords found</h3>
                <p style="color: #64748b;">Add some passwords using the form above or try a different search query.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.session_state['view_mode'] == "grid":
                # Grid view with modern cards
                num_cols = 3  # Number of columns in the grid
                rows = [filtered_data.items()] if len(filtered_data) <= num_cols else [
                    list(filtered_data.items())[i:i+num_cols] 
                    for i in range(0, len(filtered_data), num_cols)
                ]
                
                for row in rows:
                    cols = st.columns(num_cols)
                    for i, (service, details) in enumerate(row):
                        with cols[i % num_cols]:
                            # Modern card design
                            with st.container():
                                # Card header with service name and category
                                category_badge = f"""<span style="background-color: #e0e7ff; color: #4338ca; padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; margin-left: 0.5rem;">{details.get('category', '')}</span>""" if "category" in details else ""
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 0.5rem;">
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <h3 style="margin: 0; color: #1e293b; font-weight: 600; font-size: 1.25rem;">{service}</h3>
                                        {category_badge}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Username display
                                st.markdown(f"""
                                <div style="margin-bottom: 0.5rem;">
                                    <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Username/Email</p>
                                    <p style="margin: 0; color: #1e293b; font-weight: 500;">{details.get('username', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Password with show/hide toggle
                                if service in st.session_state['show_password']:
                                    password_display = details.get('password', '')
                                    st.markdown(f"""
                                    <div style="margin-bottom: 0.5rem;">
                                        <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Password</p>
                                        <div style="display: flex; align-items: center;">
                                            <p style="margin: 0; color: #1e293b; font-family: monospace; font-weight: 500; word-break: break-all;">{password_display}</p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    hide_button = st.button("Hide", key=f"hide_{service}", type="secondary", use_container_width=True)
                                    if hide_button:
                                        toggle_password_visibility(service)
                                else:
                                    password_display = "••••••••"
                                    st.markdown(f"""
                                    <div style="margin-bottom: 0.5rem;">
                                        <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Password</p>
                                        <p style="margin: 0; color: #1e293b; font-weight: 500;">{password_display}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    show_button = st.button("Show", key=f"show_{service}", type="secondary", use_container_width=True)
                                    if show_button:
                                        toggle_password_visibility(service)
                                
                                # Action buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    copy_username = st.button("Copy Username", key=f"copy_user_{service}", use_container_width=True)
                                    if copy_username:
                                        copy_to_clipboard(details.get('username', ''))
                                
                                with col2:
                                    copy_password = st.button("Copy Password", key=f"copy_pass_{service}", use_container_width=True)
                                    if copy_password:
                                        copy_to_clipboard(details.get('password', ''))
                                
                                # Display URL if available
                                if "url" in details and details["url"]:
                                    st.markdown(f"""
                                    <div style="margin-top: 0.5rem;">
                                        <a href="{details['url']}" target="_blank" style="color: #4361ee; text-decoration: none; display: flex; align-items: center; font-size: 0.875rem;">
                                            <span style="margin-right: 0.25rem;">🔗</span> Visit Website
                                        </a>
                                    </div>
                                    """, unsafe_allow_html=True)
            else:
                # List view with modern table design
                st.markdown("""
                <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Password List</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a custom table header
                st.markdown("""
                <div style="display: grid; grid-template-columns: 2fr 2fr 2fr 1fr 1fr; gap: 1rem; padding: 0.75rem 1rem; background-color: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600; color: #1e293b;">
                    <div>Service</div>
                    <div>Username</div>
                    <div>Password</div>
                    <div>Category</div>
                    <div>Actions</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create table rows for each password
                for service, details in filtered_data.items():
                    # Password display logic
                    if service in st.session_state['show_password']:
                        password_display = details.get('password', '')
                        password_button = f"""<button id="hide_{service}" onclick="document.dispatchEvent(new CustomEvent('streamlit:componentCommunication', {{detail:{{type:'streamlit:hide_{service}'}}}}))">Hide</button>"""
                    else:
                        password_display = "••••••••"
                        password_button = f"""<button id="show_{service}" onclick="document.dispatchEvent(new CustomEvent('streamlit:componentCommunication', {{detail:{{type:'streamlit:show_{service}'}}}}))">Show</button>"""
                    
                    # Category display
                    category = details.get('category', '')
                    category_display = f"""<span style="background-color: #e0e7ff; color: #4338ca; padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">{category}</span>""" if category else ""
                    
                    # Row HTML
                    st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 2fr 2fr 2fr 1fr 1fr; gap: 1rem; padding: 1rem; background-color: white; border-radius: 8px; margin-bottom: 0.5rem; align-items: center; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);">
                        <div style="font-weight: 500; color: #1e293b;">{service}</div>
                        <div style="color: #64748b; overflow: hidden; text-overflow: ellipsis;">{details.get('username', '')}</div>
                        <div style="font-family: monospace; color: #64748b;">{password_display}</div>
                        <div>{category_display}</div>
                        <div style="display: flex; gap: 0.5rem;">
                            <button id="copy_{service}" onclick="navigator.clipboard.writeText('{details.get('password', '')}'); alert('Password copied!');" style="background-color: #4361ee; color: white; border: none; border-radius: 4px; padding: 0.25rem 0.5rem; cursor: pointer;">Copy</button>
                            <button id="details_{service}" onclick="document.dispatchEvent(new CustomEvent('streamlit:componentCommunication', {{detail:{{type:'streamlit:details_{service}'}}}}))">Details</button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Handle button clicks using Streamlit components
                    show_button = st.button("Show", key=f"show_{service}", help="Show password", visible=False)
                    if show_button:
                        toggle_password_visibility(service)
                        st.rerun()
                    
                    hide_button = st.button("Hide", key=f"hide_{service}", help="Hide password", visible=False)
                    if hide_button:
                        toggle_password_visibility(service)
                        st.rerun()
                
                # Display details for selected service if any
                if 'selected_service' in st.session_state and st.session_state['selected_service'] in filtered_data:
                    service = st.session_state['selected_service']
                    details = filtered_data[service]
                    
                    st.markdown("""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin: 1.5rem 0;">
                        <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Password Details</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Service</p>
                            <p style="margin: 0; color: #1e293b; font-weight: 500; font-size: 1.25rem;">{service}</p>
                        </div>
                        
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Username/Email</p>
                            <p style="margin: 0; color: #1e293b; font-weight: 500;">{details.get('username', '')}</p>
                        </div>
                        
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Password</p>
                            <p style="margin: 0; color: #1e293b; font-family: monospace; font-weight: 500;">{details.get('password', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Category</p>
                            <p style="margin: 0; color: #1e293b; font-weight: 500;">{details.get('category', 'None')}</p>
                        </div>
                        
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Created</p>
                            <p style="margin: 0; color: #1e293b; font-weight: 500;">{details.get('created', 'Unknown')}</p>
                        </div>
                        
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Last Modified</p>
                            <p style="margin: 0; color: #1e293b; font-weight: 500;">{details.get('modified', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if "url" in details and details["url"]:
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Website URL</p>
                            <a href="{details['url']}" target="_blank" style="color: #4361ee; text-decoration: none;">{details['url']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if "notes" in details and details["notes"]:
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Notes</p>
                            <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 0.5rem; white-space: pre-wrap;">{details['notes']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Copy Username", key=f"detail_copy_user_{service}", use_container_width=True):
                            copy_to_clipboard(details.get('username', ''))
                            st.success("Username copied to clipboard!")
                    
                    with col2:
                        if st.button("Copy Password", key=f"detail_copy_pass_{service}", use_container_width=True):
                            copy_to_clipboard(details.get('password', ''))
                            st.success("Password copied to clipboard!")
                    
                    with col3:
                        if st.button("Delete", key=f"detail_delete_{service}", use_container_width=True):
                            if service in st.session_state['password_data']:
                                del st.session_state['password_data'][service]
                                if service in st.session_state['show_password']:
                                    del st.session_state['show_password'][service]
                                st.success(f"Deleted {service}")
                                save_passwords()
                                st.session_state.pop('selected_service', None)
                                st.rerun()
        
        # Action buttons at the bottom with modern design
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-top: 2rem;">
            <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save Changes", key="save_button", use_container_width=True):
                save_passwords()
                st.success("Changes saved successfully!")
        
        with col2:
            if st.button("Export as JSON", key="export_button", use_container_width=True):
                json_data = json.dumps(st.session_state['password_data'], indent=4)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="password_export.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("Refresh", key="refresh_button", use_container_width=True):
                st.rerun()
    
    elif app_mode == "Settings":
        # Modern settings page
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <h2 style="color: #1e293b; margin: 0; font-weight: 600;">Settings</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">Customize your password manager</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Change master password with modern design
        with st.expander("Change Master Password"):
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="color: #64748b; margin: 0 0 1rem 0;">Update your master password to keep your account secure. Make sure to remember your new password!</p>
            </div>
            """, unsafe_allow_html=True)
            
            old_password = st.text_input("Current Master Password", type="password", key="old_password")
            new_password = st.text_input("New Master Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm New Master Password", type="password", key="confirm_password")
            
            if new_password:
                strength, css_class = check_password_strength(new_password)
                st.markdown(f"Password Strength: <span class='{css_class}'>{strength}</span>", unsafe_allow_html=True)
            
            if st.button("Change Password", use_container_width=True):
                if old_password != st.session_state['master_password']:
                    st.error("Current password is incorrect!")
                elif new_password != confirm_password:
                    st.error("New passwords do not match!")
                elif not new_password:
                    st.error("New password cannot be empty!")
                else:
                    # Re-encrypt data with new password
                    data_backup = st.session_state['password_data'].copy()
                    st.session_state['master_password'] = new_password
                    save_encrypted_data(data_backup, new_password)
                    st.success("Master password changed successfully!")
        
        # UI Settings
        with st.expander("UI Settings"):
            default_view = st.radio(
                "Default View Mode", 
                ["Grid", "List"],
                index=0 if st.session_state['view_mode'] == "grid" else 1
            )
            st.session_state['view_mode'] = default_view.lower()
            
            st.info("More settings will be available in future updates.")
    
    elif app_mode == "Backup & Restore":
        # Modern backup & restore page
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <h2 style="color: #1e293b; margin: 0; font-weight: 600;">Backup & Restore</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">Protect your password data with backups</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Two column layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Backup section with modern design
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
                <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Create Backup</h3>
                <p style="color: #64748b; margin: 0 0 1rem 0;">Create an encrypted backup of your password database.</p>
            </div>
            """, unsafe_allow_html=True)
            
            backup_password = st.text_input("Backup Password (optional)", 
                                        help="If provided, this password will be used to encrypt the backup instead of your master password.",
                                        type="password", key="backup_password")
            
            backup_name = st.text_input("Backup Name (optional)", 
                                    placeholder="e.g., Monthly Backup",
                                    help="A name to identify this backup",
                                    key="backup_name")
            
            if st.button("Create Backup", use_container_width=True, key="create_backup"):
                password_to_use = backup_password if backup_password else st.session_state['master_password']
                
                if not os.path.exists("backups"):
                    os.makedirs("backups")
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"backups/passwords_backup_{timestamp}.db"
                
                # Add backup name to filename if provided
                if backup_name:
                    safe_name = ''.join(c if c.isalnum() else '_' for c in backup_name)
                    backup_filename = f"backups/passwords_backup_{safe_name}_{timestamp}.db"
                
                # Create backup
                if create_external_backup(st.session_state['password_data'], password_to_use, backup_filename):
                    st.success(f"Backup created successfully: {os.path.basename(backup_filename)}")
                    
                    # Offer download option
                    with open(backup_filename, "rb") as f:
                        backup_data = f.read()
                        
                    st.download_button(
                        label="Download Backup File",
                        data=backup_data,
                        file_name=os.path.basename(backup_filename),
                        mime="application/octet-stream"
                    )
                else:
                    st.error("Failed to create backup")
        
        with col2:
            # Restore section with modern design
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
                <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Restore from Backup</h3>
                <p style="color: #64748b; margin: 0 0 1rem 0;">Restore your password database from a previous backup.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to restore from file or from existing backups
            restore_option = st.radio("Restore From", ["Uploaded File", "Existing Backups"])
            
            if restore_option == "Uploaded File":
                uploaded_file = st.file_uploader("Upload Backup File", type=["db", "enc"])
                restore_password = st.text_input("Backup Password", 
                                              help="Enter the password used to encrypt this backup",
                                              type="password", key="restore_password_upload")
                
                if uploaded_file and st.button("Restore from Upload", use_container_width=True):
                    try:
                        # Save uploaded file temporarily
                        temp_file = "temp_backup.db"
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        backup_data = None
                        # Try to decrypt and load the backup
                        if uploaded_file.name.endswith('.enc'):
                            # Legacy format
                            with open(temp_file, "rb") as f:
                                encrypted_data = f.read()
                            
                            # Get salt from current database
                            conn = sqlite3.connect(st.session_state['db_path'])
                            cursor = conn.cursor()
                            cursor.execute("SELECT value FROM metadata WHERE key = 'salt'")
                            result = cursor.fetchone()
                            conn.close()
                            
                            if result:
                                salt = result[0]
                                backup_data = decrypt_data(encrypted_data, restore_password, salt)
                            else:
                                st.error("Could not retrieve salt from database")
                        else:
                            # New database format
                            backup_data = load_external_backup(restore_password, temp_file)
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                        if backup_data is not None:
                            # Confirm before overwriting
                            st.warning("This will overwrite your current passwords. Are you sure?")
                            if st.button("Yes, Restore Backup", key="confirm_restore_upload"):
                                st.session_state['password_data'] = backup_data
                                save_passwords()
                                st.success("Backup restored successfully!")
                        else:
                            st.error("Failed to decrypt backup. Incorrect password or corrupted file.")
                    
                    except Exception as e:
                        st.error(f"Failed to restore backup: {str(e)}")
                        if os.path.exists("temp_backup.db"):
                            os.remove("temp_backup.db")
            
            else:  # Existing Backups
                # List available backups
                if os.path.exists("backups"):
                    backup_files = [f for f in os.listdir("backups") if f.endswith(".db") or f.endswith(".enc")]
                    
                    if backup_files:
                        # Sort backups by date (newest first)
                        backup_files.sort(reverse=True)
                        
                        selected_backup = st.selectbox(
                            "Select Backup", 
                            backup_files,
                            format_func=lambda x: x.replace("passwords_backup_", "").replace(".db", "").replace(".enc", "").replace("_", " ")
                        )
                        
                        restore_password = st.text_input("Backup Password", 
                                                      help="Enter the password used to encrypt this backup",
                                                      type="password", key="restore_password_existing")
                        
                        if st.button("Restore Selected Backup", use_container_width=True):
                            try:
                                # Try to decrypt and load the backup
                                backup_path = os.path.join("backups", selected_backup)
                                backup_data = None
                                
                                if selected_backup.endswith('.enc'):
                                    # Legacy format
                                    with open(backup_path, "rb") as f:
                                        encrypted_data = f.read()
                                    
                                    # Get salt from current database
                                    conn = sqlite3.connect(st.session_state['db_path'])
                                    cursor = conn.cursor()
                                    cursor.execute("SELECT value FROM metadata WHERE key = 'salt'")
                                    result = cursor.fetchone()
                                    conn.close()
                                    
                                    if result:
                                        salt = result[0]
                                        backup_data = decrypt_data(encrypted_data, restore_password, salt)
                                    else:
                                        st.error("Could not retrieve salt from database")
                                else:
                                    # New database format
                                    backup_data = load_external_backup(restore_password, backup_path)
                                
                                if backup_data is not None:
                                    # Confirm before overwriting
                                    st.warning("This will overwrite your current passwords. Are you sure?")
                                    if st.button("Yes, Restore Backup", key="confirm_restore_existing"):
                                        st.session_state['password_data'] = backup_data
                                        save_passwords()
                                        st.success("Backup restored successfully!")
                                else:
                                    st.error("Failed to decrypt backup. Incorrect password or corrupted file.")
                            
                            except Exception as e:
                                st.error(f"Failed to restore backup: {str(e)}")
                                if os.path.exists("temp_backup.db"):
                                    os.remove("temp_backup.db")
                    else:
                        st.info("No backups found. Create a backup first.")
                else:
                    st.info("No backups found. Create a backup first.")
        
        # Backup management section
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin: 2rem 0 1.5rem 0;">
            <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Backup Management</h3>
            <p style="color: #64748b; margin: 0 0 1rem 0;">Manage your existing backups.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists("backups"):
            backup_files = [f for f in os.listdir("backups") if f.endswith(".db") or f.endswith(".enc")]
            
            if backup_files:
                # Create a table of backups
                st.markdown("""
                <div style="display: grid; grid-template-columns: 3fr 2fr 1fr; gap: 1rem; padding: 0.75rem 1rem; background-color: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600; color: #1e293b;">
                    <div>Backup Name</div>
                    <div>Date Created</div>
                    <div>Actions</div>
                </div>
                """, unsafe_allow_html=True)
                
                for backup_file in sorted(backup_files, reverse=True):
                    # Extract date from filename
                    date_str = backup_file.split("_")[-2] + "_" + backup_file.split("_")[-1].replace(".db", "").replace(".enc", "")
                    try:
                        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_date = "Unknown"
                    
                    # Format backup name
                    display_name = backup_file.replace("passwords_backup_", "").replace(".db", "").replace(".enc", "").replace("_", " ")
                    
                    st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 3fr 2fr 1fr; gap: 1rem; padding: 1rem; background-color: white; border-radius: 8px; margin-bottom: 0.5rem; align-items: center; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);">
                        <div style="font-weight: 500; color: #1e293b;">{display_name}</div>
                        <div style="color: #64748b;">{formatted_date}</div>
                        <div>
                            <button id="delete_{backup_file}" onclick="document.dispatchEvent(new CustomEvent('streamlit:componentCommunication', {{detail:{{type:'streamlit:delete_{backup_file}'}}}}))">Delete</button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Handle delete button
                    if st.button("Delete", key=f"delete_{backup_file}", help="Delete this backup", visible=False):
                        backup_path = os.path.join("backups", backup_file)
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                            st.success(f"Deleted backup: {backup_file}")
                            st.rerun()
            else:
                st.info("No backups found. Create a backup first.")
        else:
            st.info("No backups found. Create a backup first.")
    
    elif app_mode == "Password Generator":
        # Modern password generator page
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <h2 style="color: #1e293b; margin: 0; font-weight: 600;">Password Generator</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">Create strong, secure passwords</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Two column layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
                <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Generator Options</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Password options
            pw_length = st.slider("Password Length", min_value=8, max_value=64, value=16)
            
            # Character options
            st.markdown("### Character Types")
            use_lowercase = st.checkbox("Include Lowercase Letters (a-z)", value=True)
            use_uppercase = st.checkbox("Include Uppercase Letters (A-Z)", value=True)
            use_digits = st.checkbox("Include Digits (0-9)", value=True)
            use_special = st.checkbox("Include Special Characters (!@#$%)", value=True)
            
            # Advanced options
            with st.expander("Advanced Options"):
                exclude_similar = st.checkbox("Exclude Similar Characters (i, l, 1, L, o, 0, O)", value=False)
                exclude_ambiguous = st.checkbox("Exclude Ambiguous Characters ({}, [], (), /, \\, etc.)", value=False)
                require_all_types = st.checkbox("Require All Selected Character Types", value=True)
                custom_exclude = st.text_input("Custom Characters to Exclude")
                num_passwords = st.number_input("Number of Passwords to Generate", min_value=1, max_value=10, value=3)
            
            # Generate button
            if st.button("Generate Passwords", use_container_width=True):
                if not any([use_lowercase, use_uppercase, use_digits, use_special]):
                    st.error("Please select at least one character type.")
                else:
                    # Generate passwords
                    chars = ""
                    required_chars = []
                    
                    # Determine character sets
                    if use_lowercase:
                        lowercase_chars = string.ascii_lowercase
                        if exclude_similar:
                            lowercase_chars = lowercase_chars.replace('i', '').replace('l', '').replace('o', '')
                        chars += lowercase_chars
                        required_chars.append(random.choice(lowercase_chars))
                    
                    if use_uppercase:
                        uppercase_chars = string.ascii_uppercase
                        if exclude_similar:
                            uppercase_chars = uppercase_chars.replace('I', '').replace('L', '').replace('O', '')
                        chars += uppercase_chars
                        required_chars.append(random.choice(uppercase_chars))
                    
                    if use_digits:
                        digits = string.digits
                        if exclude_similar:
                            digits = digits.replace('0', '').replace('1', '')
                        chars += digits
                        required_chars.append(random.choice(digits))
                    
                    if use_special:
                        special_chars = string.punctuation
                        if exclude_ambiguous:
                            for c in "{}[]()/\\'\"`~,;:.<>":
                                special_chars = special_chars.replace(c, '')
                        chars += special_chars
                        required_chars.append(random.choice(special_chars))
                    
                    # Apply custom exclusions
                    if custom_exclude:
                        for c in custom_exclude:
                            chars = chars.replace(c, '')
                    
                    # Ensure there are enough characters to generate a password
                    if len(chars) < 4:
                        st.error("Not enough characters available with current settings. Please adjust your options.")
                    else:
                        # Generate passwords
                        generated_passwords = []
                        for _ in range(num_passwords):
                            if require_all_types and len(required_chars) > 0:
                                # Start with required characters
                                password = required_chars.copy()
                                
                                # Fill the rest with random characters
                                password.extend(random.choice(chars) for _ in range(pw_length - len(password)))
                                
                                # Shuffle the password
                                random.shuffle(password)
                                password = ''.join(password)
                            else:
                                # Generate a completely random password
                                password = ''.join(random.choice(chars) for _ in range(pw_length))
                            
                            generated_passwords.append(password)
                        
                        # Store in session state
                        st.session_state['generated_passwords'] = generated_passwords
        
        with col2:
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
                <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.25rem;">Generated Passwords</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display generated passwords
            if 'generated_passwords' in st.session_state and st.session_state['generated_passwords']:
                for i, password in enumerate(st.session_state['generated_passwords']):
                    # Password card
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #4361ee;">
                        <p style="font-family: monospace; margin: 0; word-break: break-all; font-size: 1rem; color: #1e293b;">
                        {password}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Password strength
                    strength, css_class = check_password_strength(password)
                    st.markdown(f"Strength: <span class='{css_class}'>{strength}</span>", unsafe_allow_html=True)
                    
                    # Copy button
                    if st.button(f"Copy to Clipboard", key=f"copy_{i}", use_container_width=True):
                        copy_to_clipboard(password)
                        st.success("Password copied to clipboard!")
                    
                    st.markdown("<hr style='margin: 1rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #f8fafc; padding: 2rem; border-radius: 10px; text-align: center; margin-top: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔑</div>
                    <h3 style="color: #1e293b; margin-bottom: 0.5rem; font-weight: 600;">No Passwords Generated</h3>
                    <p style="color: #64748b;">Configure your options and click 'Generate Passwords' to create secure passwords.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Password tips
            st.markdown("""
            <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #0ea5e9;">
                <h4 style="color: #0c4a6e; margin: 0 0 0.5rem 0; font-weight: 600;">Password Tips</h4>
                <ul style="color: #0c4a6e; margin: 0; padding-left: 1.5rem;">
                    <li>Use at least 12 characters for better security</li>
                    <li>Include a mix of character types</li>
                    <li>Avoid using personal information</li>
                    <li>Use a different password for each account</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Add statistics dashboard at the bottom of the app
if st.session_state['authenticated'] and app_mode == "Password Manager":
    with st.expander("Password Statistics", expanded=False):
        if not st.session_state['password_data']:
            st.info("No passwords to analyze. Add some passwords to see statistics.")
        else:
            # Calculate statistics
            total_passwords = len(st.session_state['password_data'])
            
            # Password strength distribution
            strength_counts = {"High": 0, "Medium": 0, "Low": 0}
            for service, details in st.session_state['password_data'].items():
                if "password" in details:
                    strength, _ = check_password_strength(details["password"])
                    strength_counts[strength] += 1
            
            # Category distribution
            categories = {}
            for service, details in st.session_state['password_data'].items():
                category = details.get("category", "Uncategorized")
                categories[category] = categories.get(category, 0) + 1
            
            # Password length distribution
            length_ranges = {"Short (1-8)": 0, "Medium (9-16)": 0, "Long (17+)": 0}
            for service, details in st.session_state['password_data'].items():
                password = details.get("password", "")
                if len(password) <= 8:
                    length_ranges["Short (1-8)"] += 1
                elif len(password) <= 16:
                    length_ranges["Medium (9-16)"] += 1
                else:
                    length_ranges["Long (17+)"] += 1
            
            # Password reuse
            password_counts = {}
            for service, details in st.session_state['password_data'].items():
                password = details.get("password", "")
                if password:
                    password_counts[password] = password_counts.get(password, 0) + 1
            
            reused_passwords = sum(1 for count in password_counts.values() if count > 1)
            reused_percentage = reused_passwords / total_passwords * 100 if total_passwords > 0 else 0
            
            # Display statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary")
                st.metric("Total Passwords", total_passwords)
                st.metric("Reused Passwords", f"{reused_passwords} ({reused_percentage:.1f}%)")
                
                st.subheader("Password Strength")
                for strength, count in strength_counts.items():
                    percentage = count / total_passwords * 100 if total_passwords > 0 else 0
                    st.text(f"{strength}: {count} ({percentage:.1f}%)")
                
                # Progress bars for strength distribution
                if total_passwords > 0:
                    st.progress(strength_counts["High"] / total_passwords)
                    st.progress(strength_counts["Medium"] / total_passwords)
                    st.progress(strength_counts["Low"] / total_passwords)
            
            with col2:
                st.subheader("Categories")
                for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / total_passwords * 100
                    st.text(f"{category}: {count} ({percentage:.1f}%)")
                
                st.subheader("Password Length")
                for length_range, count in length_ranges.items():
                    percentage = count / total_passwords * 100 if total_passwords > 0 else 0
                    st.text(f"{length_range}: {count} ({percentage:.1f}%)")
            
            # Security recommendations
            st.subheader("Security Recommendations")
            recommendations = []
            
            if strength_counts["Low"] > 0:
                recommendations.append(f"- Upgrade {strength_counts['Low']} weak passwords to improve security.")
            
            if reused_passwords > 0:
                recommendations.append(f"- Replace {reused_passwords} reused passwords with unique ones.")
            
            if length_ranges["Short (1-8)"] > 0:
                recommendations.append(f"- Increase the length of {length_ranges['Short (1-8)']} short passwords.")
            
            # Check for passwords not updated in 90 days
            old_passwords = 0
            for service, details in st.session_state['password_data'].items():
                if "modified" in details:
                    try:
                        modified_date = datetime.datetime.strptime(details["modified"], "%Y-%m-%d %H:%M:%S")
                        days_old = (datetime.datetime.now() - modified_date).days
                        if days_old > 90:
                            old_passwords += 1
                    except (ValueError, TypeError):
                        pass
            
            if old_passwords > 0:
                recommendations.append(f"- Update {old_passwords} passwords that haven't been changed in 90+ days.")
            
            if recommendations:
                for recommendation in recommendations:
                    st.markdown(recommendation)
            else:
                st.success("Great job! Your password vault looks secure.")

# Register this app to run on Streamlit
if __name__ == "__main__":
    # Clear query parameters
    st.query_params.clear()

def create_external_backup(data, password, backup_path):
    """Create an external backup of the database"""
    # Generate a new salt for this backup
    salt = os.urandom(16)
    
    # Add last modified timestamp
    data_with_metadata = {
        "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passwords": data
    }
    
    encrypted_data = encrypt_data(data_with_metadata, password, salt)
    
    # Create a new SQLite database for the backup
    conn = sqlite3.connect(backup_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value BLOB
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS passwords (
        id INTEGER PRIMARY KEY,
        encrypted_data BLOB,
        created_at TEXT,
        last_modified TEXT
    )
    ''')
    
    # Store salt
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ('salt', salt))
    
    # Store encrypted data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO passwords (encrypted_data, created_at, last_modified) VALUES (?, ?, ?)",
        (encrypted_data, timestamp, timestamp)
    )
    
    conn.commit()
    conn.close()
    
    return True

def load_external_backup(password, backup_path):
    """Load data from an external backup database"""
    try:
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='metadata' OR name='passwords')")
        tables = cursor.fetchall()
        if len(tables) < 2:
            conn.close()
            return None
        
        # Get salt
        cursor.execute("SELECT value FROM metadata WHERE key = 'salt'")
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
            
        salt = result[0]
        
        # Get password data
        cursor.execute("SELECT encrypted_data FROM passwords WHERE last_modified NOT LIKE 'BACKUP_%' ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
            
        encrypted_data = result[0]
        conn.close()
        
        decrypted_data = decrypt_data(encrypted_data, password, salt)
        
        if decrypted_data is None:
            return None
        
        # Handle both new format (with metadata) and old format
        if isinstance(decrypted_data, dict) and "passwords" in decrypted_data:
            return decrypted_data["passwords"]
        else:
            return decrypted_data
            
    except Exception as e:
        st.error(f"Error loading backup database: {str(e)}")
        return None