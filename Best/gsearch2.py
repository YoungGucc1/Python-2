import os
import requests
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext

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

def save_images(image_items, save_directory, query):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    query_directory = os.path.join(save_directory, query.replace(" ", "_"))
    if not os.path.exists(query_directory):
        os.makedirs(query_directory)
    
    for i, item in enumerate(image_items):
        image_url = item['link']
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_name = f"image_{i+1}.jpg"
                image_path = os.path.join(query_directory, image_name)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved {image_name} for query '{query}'")
            else:
                print(f"Failed to download {image_url} for query '{query}'")
        except Exception as e:
            print(f"Error downloading {image_url} for query '{query}': {e}")

def on_search():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    
    save_directory = filedialog.askdirectory()
    if not save_directory:
        return
    
    with open(file_path, 'r') as file:
        queries = file.read().splitlines()
    
    output_text.delete(1.0, tk.END)  # Clear previous output
    
    for query in queries:
        if query.strip():
            output_text.insert(tk.END, f"Searching for: {query}\n")
            image_items = search_images(query)
            if image_items:
                save_images(image_items, save_directory, query)
                output_text.insert(tk.END, f"Images for '{query}' saved to {save_directory}\n")
            else:
                output_text.insert(tk.END, f"No images found for '{query}'\n")
    
    output_text.insert(tk.END, "Batch search completed.\n")

# GUI Setup
root = tk.Tk()
root.title("Batch Image Search and Save")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Select a file with search queries:")
label.grid(row=0, column=0, padx=5, pady=5)

search_button = tk.Button(frame, text="Search and Save Images", command=on_search)
search_button.grid(row=1, column=0, columnspan=2, pady=10)

output_text = scrolledtext.ScrolledText(frame, width=80, height=20)
output_text.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()