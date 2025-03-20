import os
import requests
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from tkinter import ttk
import threading
import time

# Replace with your Google API key and Custom Search Engine ID
API_KEY = 'AIzaSyDKkI4mO8zZ_XgUcCSksUvR1wSrIQ1bWck'
CSE_ID = "140ad6cab866c40cf"

# Global variables for controlling the search
is_paused = False
is_stopped = False

def search_images(query, num_results=10):
    url = f"https://customsearch.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': CSE_ID,
        'key': API_KEY,
        'searchType': 'image',
        'num': num_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        messagebox.showerror("Error", f"Failed to fetch images: {response.status_code}")
        return []

def save_images(image_items, save_directory, query):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    query_directory = os.path.join(save_directory, query.replace(" ", "_"))
    if not os.path.exists(query_directory):
        os.makedirs(query_directory)
    
    for i, item in enumerate(image_items):
        if is_stopped:
            break
        while is_paused:
            time.sleep(1)
        image_url = item['link']
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_name = f"{query.replace(' ', '_')}_{i+1}.jpg"
                image_path = os.path.join(query_directory, image_name)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                log(f"Saved {image_name} for query '{query}'")
            else:
                log(f"Failed to download {image_url} for query '{query}'")
        except Exception as e:
            log(f"Error downloading {image_url} for query '{query}': {e}")

def on_search():
    global is_paused, is_stopped
    is_paused = False
    is_stopped = False

    save_directory = filedialog.askdirectory()
    if not save_directory:
        return
    
    queries = query_text.get("1.0", tk.END).splitlines()
    
    num_images = int(num_images_entry.get())
    
    output_text.delete(1.0, tk.END)  # Clear previous output
    
    progress['maximum'] = len(queries)
    progress['value'] = 0
    
    def search_thread():
        for query in queries:
            if is_stopped:
                break
            if query.strip():
                log(f"Searching for: {query}")
                image_items = search_images(query, num_images)
                if image_items:
                    save_images(image_items, save_directory, query)
                    log(f"Images for '{query}' saved to {save_directory}")
                else:
                    log(f"No images found for '{query}'")
            progress['value'] += 1
            root.update_idletasks()
        log("Batch search completed.")
    
    threading.Thread(target=search_thread).start()

def on_pause():
    global is_paused
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")

def on_stop():
    global is_stopped
    is_stopped = True

def log(message):
    output_text.insert(tk.END, message + "\n")
    output_text.see(tk.END)

# GUI Setup
root = tk.Tk()
root.title("Batch Image Search and Save")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Enter search queries (one per line):")
label.grid(row=0, column=0, padx=5, pady=5)

query_text = tk.Text(frame, width=50, height=10)
query_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

num_images_label = tk.Label(frame, text="Number of images to download:")
num_images_label.grid(row=2, column=0, padx=5, pady=5)

num_images_entry = tk.Entry(frame)
num_images_entry.grid(row=2, column=1, padx=5, pady=5)
num_images_entry.insert(0, "10")

search_button = tk.Button(frame, text="Search and Save Images", command=on_search)
search_button.grid(row=3, column=0, columnspan=2, pady=10)

pause_button = tk.Button(frame, text="Pause", command=on_pause)
pause_button.grid(row=4, column=0, pady=10)

stop_button = tk.Button(frame, text="Stop", command=on_stop)
stop_button.grid(row=4, column=1, pady=10)

progress = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress.grid(row=5, column=0, columnspan=2, pady=10)

output_text = scrolledtext.ScrolledText(frame, width=80, height=20)
output_text.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()