import os
import requests
import tkinter as tk
from tkinter import messagebox, filedialog

# Replace with your Google API key and Custom Search Engine ID
API_KEY = 'AIzaSyDKkI4mO8zZ_XgUcCSksUvR1wSrIQ1bWck'
CSE_ID = "140ad6cab866c40cf"

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

def save_images(image_items, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for i, item in enumerate(image_items):
        image_url = item['link']
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_name = f"image_{i+1}.jpg"
                image_path = os.path.join(save_directory, image_name)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved {image_name}")
            else:
                print(f"Failed to download {image_url}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")

def on_search():
    query = entry.get()
    if not query:
        messagebox.showwarning("Warning", "Please enter a search keyword.")
        return
    
    save_directory = filedialog.askdirectory()
    if not save_directory:
        return
    
    image_items = search_images(query)
    if image_items:
        save_images(image_items, save_directory)
        messagebox.showinfo("Success", f"Images saved to {save_directory}")
    else:
        messagebox.showinfo("Info", "No images found.")

# GUI Setup
root = tk.Tk()
root.title("Image Search and Save")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Enter search keyword:")
label.grid(row=0, column=0, padx=5, pady=5)

entry = tk.Entry(frame, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

search_button = tk.Button(frame, text="Search and Save Images", command=on_search)
search_button.grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()