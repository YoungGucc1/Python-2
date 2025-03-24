import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import requests
from bs4 import BeautifulSoup
import os
import re
import threading
import time
import queue
import pyperclip
from PIL import Image, ImageTk, ImageFilter
from io import BytesIO

# Global variables
running = False
paused = False
capture_mode = False
clipboard_history = set()
link_process_thread = None
download_thread = None
clipboard_thread = None
download_queue = queue.Queue()
link_queue = queue.Queue()
processed_links = []

# Color scheme - Modern dark theme
COLORS = {
    "bg_dark": "#1e1e2e",       # Dark background
    "bg_medium": "#313244",     # Medium background
    "bg_light": "#45475a",      # Light background
    "accent": "#89b4fa",        # Blue accent
    "accent_green": "#a6e3a1",  # Green accent
    "accent_red": "#f38ba8",    # Red accent
    "accent_yellow": "#f9e2af", # Yellow accent
    "text": "#cdd6f4",          # Light text
    "text_dim": "#9399b2"       # Dimmed text
}

# Default headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_links(url):
    """Extract image links from a URL"""
    try:
        # Add prefix if needed
        if not url.startswith(("http://", "https://")) and prefix_entry.get():
            url = prefix_entry.get().strip() + url
            
        log_message(f"Processing: {url}", "info")
        
        # Make sure URL is properly formatted
        if "goodfon.ru" in url and not url.startswith(("http://", "https://")):
            url = "https://www.goodfon.ru" + url
            
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find image links - adjust selectors based on the website structure
        links = []
        
        # For goodfon.ru - look for download links
        if "goodfon.ru" in url:
            # First try to find the download buttons
            download_links = soup.find_all('a', class_='wallpaper__download__rbut')
            
            if download_links:
                for link in download_links:
                    if 'href' in link.attrs:
                        href = link['href']
                        if not href.startswith(('http://', 'https://')):
                            href = 'https://www.goodfon.ru' + href
                        links.append(href)
                        log_message(f"Found download link: {href}", "success")
            
            # If no download links, try to find the main image
            if not links:
                main_img = soup.find('img', class_='wallpaper__item__fon__img')
                if main_img and 'src' in main_img.attrs:
                    src = main_img['src']
                    if not src.startswith(('http://', 'https://')):
                        if src.startswith('//'):
                            src = 'https:' + src
                        else:
                            src = 'https://www.goodfon.ru' + src.lstrip('/')
                    links.append(src)
                    log_message(f"Found main image: {src}", "success")
        
        # Generic image finder for other sites
        if not links:
            for img in soup.find_all('img', {'src': True}):
                src = img['src']
                if re.search(r'\.(jpg|jpeg|png|gif)(\?|$)', src, re.I):
                    if not src.startswith(('http://', 'https://')):
                        if src.startswith('//'):
                            src = 'https:' + src
                        else:
                            base_url = url.split('/')[0] + '//' + url.split('/')[2]
                            src = base_url + '/' + src.lstrip('/')
                    links.append(src)
        
        log_message(f"Found {len(links)} links", "info")
        return links
    except Exception as e:
        log_message(f"Error processing {url}: {str(e)}", "error")
        return []

def link_processor():
    """Thread function to process URLs and extract image links"""
    global processed_links
    
    while running and not paused:
        try:
            # Get a URL from the queue
            url = link_queue.get(timeout=1)
            
            if url is None:  # Sentinel value to stop the thread
                break
                
            # Extract links from the URL
            links = extract_links(url)
            
            # Add links to the processed links list
            if links:
                processed_links.extend(links)
                update_status_label(f"Found {len(processed_links)} images")
            
            # Mark this task as done
            link_queue.task_done()
            
        except queue.Empty:
            # Check if we've processed all URLs
            if link_queue.empty():
                log_message("URL processing complete", "success")
                # Start downloading if we have links
                if processed_links and running and not paused:
                    root.after(100, start_downloads)  # Use after to safely call from thread
                break
            continue
        except Exception as e:
            log_message(f"Link processor error: {str(e)}", "error")
            time.sleep(0.5)

def start_downloads():
    """Start downloading the processed links"""
    global download_thread
    
    if not processed_links:
        log_message("No images to download", "warning")
        return
    
    log_message(f"Starting download of {len(processed_links)} images", "success")
    
    # Reset progress bar
    progress_bar["maximum"] = len(processed_links)
    progress_bar["value"] = 0
    
    # Add all links to the download queue
    for link in processed_links:
        download_queue.put(link)
    
    # Start the download thread if not already running
    if download_thread is None or not download_thread.is_alive():
        download_thread = threading.Thread(target=download_worker)
        download_thread.daemon = True
        download_thread.start()
        log_message("Download thread started", "info")

def download_worker():
    """Thread function to download images"""
    download_count = 0
    total_downloads = download_queue.qsize()
    
    log_message(f"Download worker started. Queue size: {total_downloads}", "info")
    
    if total_downloads == 0:
        log_message("Error: Download queue is empty", "error")
        return
    
    while running and not paused:
        try:
            # Get a URL from the queue with a timeout
            try:
                url = download_queue.get(timeout=1)
            except queue.Empty:
                # Check if we've downloaded everything
                if download_count >= total_downloads:
                    log_message("All downloads completed!", "success")
                    update_status_label("Downloads complete")
                    break
                continue
            
            if url is None:  # Sentinel value to stop the thread
                break
            
            log_message(f"Downloading: {url}", "info")
            
            try:
                # Download the image
                response = requests.get(url, headers=HEADERS, stream=True, timeout=15)
                response.raise_for_status()
                
                # Extract filename from URL
                filename = os.path.basename(url).split('?')[0]
                if not filename or not re.search(r'\.(jpg|jpeg|png|gif)$', filename, re.I):
                    filename = f"image_{download_count + 1}.jpg"
                
                # Create downloads directory if it doesn't exist
                os.makedirs("downloads", exist_ok=True)
                filepath = os.path.join("downloads", filename)
                
                # Save the image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk and running and not paused:
                            f.write(chunk)
                
                # Update progress
                download_count += 1
                progress_bar["value"] = download_count
                update_status_label(f"Downloaded {download_count}/{total_downloads}")
                
                # Show thumbnail in the preview area
                try:
                    show_thumbnail(filepath)
                except Exception as e:
                    log_message(f"Error showing thumbnail: {str(e)}", "error")
                
                log_message(f"Downloaded: {filename}", "success")
                
            except Exception as e:
                log_message(f"Error downloading {url}: {str(e)}", "error")
            
            # Mark this task as done
            download_queue.task_done()
            
        except Exception as e:
            log_message(f"Download worker error: {str(e)}", "error")
            time.sleep(0.5)
    
    # Enable the start button when done
    if not running or download_count >= total_downloads:
        root.after(0, lambda: toggle_buttons(False))

def show_thumbnail(image_path):
    """Show a thumbnail of the downloaded image"""
    try:
        # Open the image and create a thumbnail
        img = Image.open(image_path)
        img.thumbnail((100, 100))
        
        # Add a subtle blur effect for aesthetics
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Update the preview label
        preview_label.config(image=photo)
        preview_label.image = photo  # Keep a reference to prevent garbage collection
    except Exception as e:
        log_message(f"Error creating thumbnail: {str(e)}", "error")

def start_process():
    """Start the link processing and download process"""
    global running, paused, processed_links, link_process_thread
    
    if running:
        return
    
    # Get URLs from the text area
    urls = url_text.get("1.0", tk.END).strip().split('\n')
    urls = [url.strip() for url in urls if url.strip()]
    
    if not urls:
        log_message("Please enter at least one URL", "warning")
        return
    
    # Reset state
    running = True
    paused = False
    processed_links = []
    
    # Clear queues
    clear_queue(link_queue)
    clear_queue(download_queue)
    
    # Update UI
    toggle_buttons(True)
    update_status_label("Processing URLs...")
    log_message(f"Processing {len(urls)} URLs", "info")
    
    # Add URLs to the link queue
    for url in urls:
        link_queue.put(url)
    
    # Start the link processing thread
    if link_process_thread is not None and link_process_thread.is_alive():
        log_message("Warning: Link processor thread already running", "warning")
    else:
        link_process_thread = threading.Thread(target=link_processor)
        link_process_thread.daemon = True
        link_process_thread.start()
        log_message("Link processor thread started", "info")

def stop_process():
    """Stop all processing"""
    global running
    
    running = False
    log_message("Stopping all processes...", "warning")
    
    # Clear queues
    clear_queue(link_queue)
    clear_queue(download_queue)
    
    # Update UI
    toggle_buttons(False)
    update_status_label("Stopped")

def clear_queue(q):
    """Clear a queue safely"""
    try:
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except:
        pass

def pause_process():
    """Pause or resume processing"""
    global paused
    
    paused = not paused
    
    if paused:
        log_message("Process paused", "warning")
        pause_button.config(text="Resume", bg=COLORS["accent_green"])
        update_status_label("Paused")
    else:
        log_message("Process resumed", "success")
        pause_button.config(text="Pause", bg=COLORS["accent_yellow"])
        update_status_label("Resuming...")

def toggle_buttons(is_running):
    """Update button states based on running state"""
    if is_running:
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        pause_button.config(state=tk.NORMAL)
        save_button.config(state=tk.DISABLED)
        load_button.config(state=tk.DISABLED)
    else:
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        pause_button.config(state=tk.DISABLED, text="Pause", bg=COLORS["accent_yellow"])
        save_button.config(state=tk.NORMAL)
        load_button.config(state=tk.NORMAL)

def update_status_label(text):
    """Update the status label"""
    status_var.set(text)
    root.update_idletasks()

def log_message(message, level="info"):
    """Add a message to the log with timestamp and color"""
    timestamp = time.strftime("%H:%M:%S")
    
    # Set tag based on message level
    if level == "error":
        tag = "error"
    elif level == "warning":
        tag = "warning"
    elif level == "success":
        tag = "success"
    else:
        tag = "info"
    
    # Insert message with tag
    log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
    log_text.see(tk.END)  # Auto-scroll to the latest message

def toggle_capture_mode():
    """Toggle clipboard capture mode"""
    global capture_mode, clipboard_thread
    
    capture_mode = not capture_mode
    
    if capture_mode:
        capture_button.config(text="Disable Capture", bg=COLORS["accent_red"])
        log_message("Clipboard capture enabled", "success")
        
        # Start clipboard monitoring thread
        clipboard_thread = threading.Thread(target=monitor_clipboard)
        clipboard_thread.daemon = True
        clipboard_thread.start()
    else:
        capture_button.config(text="Enable Capture", bg=COLORS["accent"])
        log_message("Clipboard capture disabled", "warning")

def monitor_clipboard():
    """Monitor clipboard for URLs"""
    global capture_mode
    
    last_value = ""
    
    while capture_mode:
        try:
            current_value = pyperclip.paste()
            
            # Check if clipboard content changed and is a URL
            if (current_value != last_value and 
                current_value.startswith(("http://", "https://")) and 
                current_value not in clipboard_history):
                
                clipboard_history.add(current_value)
                
                # Add to text widget
                url_text.insert(tk.END, current_value + "\n")
                log_message(f"Captured: {current_value}", "success")
                
                # Update last value
                last_value = current_value
                
        except Exception as e:
            log_message(f"Clipboard error: {str(e)}", "error")
            
        time.sleep(0.5)

def save_url_list():
    """Save the URL list to a file"""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Save URL List"
    )
    
    if not file_path:
        return
    
    try:
        with open(file_path, 'w') as f:
            f.write(url_text.get("1.0", tk.END))
        log_message(f"URLs saved to {file_path}", "success")
    except Exception as e:
        log_message(f"Error saving URLs: {str(e)}", "error")

def load_url_list():
    """Load URLs from a file"""
    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Load URL List"
    )
    
    if not file_path:
        return
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Clear and set new content
        url_text.delete("1.0", tk.END)
        url_text.insert("1.0", content)
        log_message(f"URLs loaded from {file_path}", "success")
    except Exception as e:
        log_message(f"Error loading URLs: {str(e)}", "error")

def create_tooltip(widget, text):
    """Create a tooltip for a widget"""
    def enter(event):
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        
        # Create a toplevel window
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tooltip, text=text, justify=tk.LEFT,
                         background=COLORS["bg_dark"], foreground=COLORS["text"],
                         relief="solid", borderwidth=1, padx=5, pady=2)
        label.pack(ipadx=1)
        
        widget.tooltip = tooltip
        
    def leave(event):
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
    
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

def on_closing():
    """Handle window closing"""
    global running, capture_mode
    
    if running:
        if messagebox.askyesno("Quit", "Downloads are in progress. Are you sure you want to quit?"):
            running = False
            capture_mode = False
            root.destroy()
    else:
        root.destroy()

def debug_info():
    """Display debug information about the current state"""
    info = [
        f"Processed Links: {len(processed_links)}",
        f"Download Queue Size: {download_queue.qsize()}",
        f"Link Queue Size: {link_queue.qsize()}",
        f"Running: {running}",
        f"Paused: {paused}",
        f"Capture Mode: {capture_mode}"
    ]
    
    if processed_links:
        info.append("\nFirst 3 processed links:")
        for i, link in enumerate(processed_links[:3]):
            info.append(f"{i+1}. {link}")
    
    log_message("\n".join(info), "info")
    
    # Try to download the first link directly as a test
    if processed_links and running:
        test_url = processed_links[0]
        log_message(f"Testing direct download of: {test_url}", "warning")
        try:
            response = requests.get(test_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                log_message(f"Test download successful! Content type: {response.headers.get('content-type')}", "success")
                
                # Save test file
                test_file = os.path.join("downloads", "test_download.jpg")
                os.makedirs("downloads", exist_ok=True)
                with open(test_file, 'wb') as f:
                    f.write(response.content)
                log_message(f"Test file saved to: {test_file}", "success")
                
                # Show thumbnail
                show_thumbnail(test_file)
            else:
                log_message(f"Test download failed with status code: {response.status_code}", "error")
        except Exception as e:
            log_message(f"Test download error: {str(e)}", "error")

# Create the main window
root = tk.Tk()
root.title("Wallpaper Downloader")
root.configure(bg=COLORS["bg_dark"])
root.geometry("800x600")
root.minsize(600, 500)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Configure styles
style = ttk.Style()
style.theme_use('default')
style.configure("TProgressbar", 
                background=COLORS["accent_green"],
                troughcolor=COLORS["bg_light"],
                borderwidth=0,
                thickness=10)

# Create a main frame with padding
main_frame = tk.Frame(root, bg=COLORS["bg_dark"], padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Top frame for URL input and controls
top_frame = tk.Frame(main_frame, bg=COLORS["bg_dark"])
top_frame.pack(fill=tk.X, pady=(0, 10))

# URL input frame
url_frame = tk.LabelFrame(top_frame, text="URLs", bg=COLORS["bg_medium"], fg=COLORS["accent"], 
                         padx=5, pady=5, font=("Segoe UI", 9, "bold"))
url_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

url_text = scrolledtext.ScrolledText(url_frame, width=40, height=8, bg=COLORS["bg_light"], 
                                    fg=COLORS["text"], insertbackground=COLORS["accent"])
url_text.pack(fill=tk.BOTH, expand=True)

# Controls frame
controls_frame = tk.LabelFrame(top_frame, text="Controls", bg=COLORS["bg_medium"], fg=COLORS["accent"], 
                              padx=5, pady=5, font=("Segoe UI", 9, "bold"))
controls_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

# Prefix input
prefix_frame = tk.Frame(controls_frame, bg=COLORS["bg_medium"])
prefix_frame.pack(fill=tk.X, pady=(0, 5))

prefix_label = tk.Label(prefix_frame, text="Prefix:", bg=COLORS["bg_medium"], fg=COLORS["text"])
prefix_label.pack(side=tk.LEFT, padx=(0, 5))

prefix_entry = tk.Entry(prefix_frame, bg=COLORS["bg_light"], fg=COLORS["text"], 
                       insertbackground=COLORS["accent"])
prefix_entry.insert(0, "https://www.goodfon.ru")
prefix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

# Status display
status_var = tk.StringVar()
status_var.set("Ready")
status_label = tk.Label(controls_frame, textvariable=status_var, bg=COLORS["bg_medium"], 
                       fg=COLORS["accent"], anchor="w")
status_label.pack(fill=tk.X, pady=(0, 5))

# Preview image
preview_frame = tk.Frame(controls_frame, bg=COLORS["bg_medium"])
preview_frame.pack(fill=tk.X, pady=(0, 5))

preview_label = tk.Label(preview_frame, bg=COLORS["bg_light"], width=10, height=5)
preview_label.pack(side=tk.LEFT, padx=(0, 5))

# Buttons frame
buttons_frame = tk.Frame(controls_frame, bg=COLORS["bg_medium"])
buttons_frame.pack(fill=tk.X)

# Action buttons
button_style = {"font": ("Segoe UI", 9, "bold"), "borderwidth": 0, 
               "relief": tk.FLAT, "padx": 10, "pady": 3}

start_button = tk.Button(buttons_frame, text="Start", bg=COLORS["accent_green"], fg=COLORS["bg_dark"], 
                        command=start_process, **button_style)
start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), pady=(0, 2))
create_tooltip(start_button, "Start processing URLs")

pause_button = tk.Button(buttons_frame, text="Pause", bg=COLORS["accent_yellow"], fg=COLORS["bg_dark"], 
                        command=pause_process, state=tk.DISABLED, **button_style)
pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2), pady=(0, 2))
create_tooltip(pause_button, "Pause/Resume processing")

stop_button = tk.Button(buttons_frame, text="Stop", bg=COLORS["accent_red"], fg=COLORS["bg_dark"], 
                       command=stop_process, state=tk.DISABLED, **button_style)
stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), pady=(0, 2))
create_tooltip(stop_button, "Stop all processing")

# File and capture buttons
file_buttons_frame = tk.Frame(buttons_frame, bg=COLORS["bg_medium"])
file_buttons_frame.pack(fill=tk.X, pady=(2, 0))

save_button = tk.Button(file_buttons_frame, text="Save List", bg=COLORS["accent"], fg=COLORS["bg_dark"], 
                       command=save_url_list, **button_style)
save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
create_tooltip(save_button, "Save URL list to file")

load_button = tk.Button(file_buttons_frame, text="Load List", bg=COLORS["accent"], fg=COLORS["bg_dark"], 
                       command=load_url_list, **button_style)
load_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
create_tooltip(load_button, "Load URL list from file")

capture_button = tk.Button(file_buttons_frame, text="Enable Capture", bg=COLORS["accent"], 
                          fg=COLORS["bg_dark"], command=toggle_capture_mode, **button_style)
capture_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
create_tooltip(capture_button, "Enable/Disable clipboard URL capture")

# Debug button
debug_button = tk.Button(file_buttons_frame, text="Debug", bg=COLORS["bg_light"], 
                       fg=COLORS["text"], command=debug_info, **button_style)
debug_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
create_tooltip(debug_button, "Show debug information")

# Progress bar
progress_frame = tk.Frame(main_frame, bg=COLORS["bg_dark"], height=20)
progress_frame.pack(fill=tk.X, pady=(0, 10))

progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode="determinate", style="TProgressbar")
progress_bar.pack(fill=tk.X)

# Log area
log_frame = tk.LabelFrame(main_frame, text="Log", bg=COLORS["bg_medium"], fg=COLORS["accent"], 
                         padx=5, pady=5, font=("Segoe UI", 9, "bold"))
log_frame.pack(fill=tk.BOTH, expand=True)

log_text = scrolledtext.ScrolledText(log_frame, bg=COLORS["bg_light"], fg=COLORS["text"], 
                                    insertbackground=COLORS["accent"])
log_text.pack(fill=tk.BOTH, expand=True)

# Configure log text tags
log_text.tag_configure("error", foreground=COLORS["accent_red"])
log_text.tag_configure("warning", foreground=COLORS["accent_yellow"])
log_text.tag_configure("success", foreground=COLORS["accent_green"])
log_text.tag_configure("info", foreground=COLORS["text"])

# Initial log message
log_message("Application started. Ready to download wallpapers.", "info")

# Start the main loop
root.mainloop()