import asyncio
import os
from datetime import datetime
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto
import tkinter as tk
from tkinter import ttk, filedialog
import threading

max_length = 250

class ImageDownloader:
    def __init__(self):
        self.running = False
        self.paused = False
        self.count = 0
        self.save_folder = ""
        self.channel_username = ""
        self.api_id = ""
        self.api_hash = ""
        self.phone_number = ""
        self.client = None

    async def download_images(self):
        self.client = TelegramClient('session', self.api_id, self.api_hash)
        async with self.client:
            await self.client.start(phone=self.phone_number)
            
            channel = await self.client.get_entity(self.channel_username)
            
            async for message in self.client.iter_messages(channel):
                if not self.running:
                    break
                
                if self.paused:
                    await asyncio.sleep(1)
                    continue
                
                if message.media and isinstance(message.media, MessageMediaPhoto):
                    caption = message.message if message.message else "no_description"
                    date = message.date.strftime("%Y%m%d_%H%M")
                    filename = f"{date}_{caption}.jpg"
                    filename = "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).rstrip()
                    if len(filename) > max_length:
                            filename = filename[:max_length]
                    full_path = os.path.join(self.save_folder, filename)
                    
                    await self.client.download_media(message.media, file=full_path)
                    self.count += 1
                    update_count(self.count)

downloader = ImageDownloader()

def start_download():
    if not all([downloader.save_folder, downloader.channel_username, 
                downloader.api_id, downloader.api_hash, downloader.phone_number]):
        status_label.config(text="Please fill in all fields and select a folder.")
        return
    downloader.running = True
    downloader.paused = False
    status_label.config(text="Downloading...")
    threading.Thread(target=lambda: asyncio.run(downloader.download_images())).start()

def stop_download():
    downloader.running = False
    downloader.paused = False
    status_label.config(text="Stopped.")

def pause_download():
    downloader.paused = not downloader.paused
    status_label.config(text="Paused." if downloader.paused else "Resumed.")

def update_count(count):
    count_label.config(text=f"Images saved: {count}")

def select_folder():
    folder = filedialog.askdirectory()
    if folder:
        downloader.save_folder = folder
        folder_label.config(text=f"Save folder: {folder}")

def update_settings():
    downloader.channel_username = channel_entry.get()
    downloader.api_id = api_id_entry.get()
    downloader.api_hash = api_hash_entry.get()
    downloader.phone_number = phone_entry.get()
    status_label.config(text="Settings updated.")

# GUI setup
root = tk.Tk()
root.title("Telegram Image Downloader")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# API ID input
api_id_label = ttk.Label(frame, text="API ID:")
api_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
api_id_entry = ttk.Entry(frame)
api_id_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
api_id_entry.insert(0, '20136831')

# API Hash input
api_hash_label = ttk.Label(frame, text="API Hash:")
api_hash_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
api_hash_entry = ttk.Entry(frame)
api_hash_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
api_hash_entry.insert(0, '6fcd67491e7d65b1c75967b97bbc97bd')

# Phone number input
phone_label = ttk.Label(frame, text="Phone Number:")
phone_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
phone_entry = ttk.Entry(frame)
phone_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
phone_entry.insert(0, '+77072971894')

# Channel username input
channel_label = ttk.Label(frame, text="Channel Username:")
channel_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
channel_entry = ttk.Entry(frame)
channel_entry.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

# Update settings button
update_button = ttk.Button(frame, text="Update Settings", command=update_settings)
update_button.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

# Folder selection
folder_button = ttk.Button(frame, text="Select Folder", command=select_folder)
folder_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

folder_label = ttk.Label(frame, text="No folder selected")
folder_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

# Control buttons
start_button = ttk.Button(frame, text="Start", command=start_download)
start_button.grid(row=7, column=0, padx=5, pady=5)

stop_button = ttk.Button(frame, text="Stop", command=stop_download)
stop_button.grid(row=7, column=1, padx=5, pady=5)

pause_button = ttk.Button(frame, text="Pause/Resume", command=pause_download)
pause_button.grid(row=7, column=2, padx=5, pady=5)

# Status and count labels
status_label = ttk.Label(frame, text="Ready.")
status_label.grid(row=8, column=0, columnspan=3, pady=10)

count_label = ttk.Label(frame, text="Images saved: 0")
count_label.grid(row=9, column=0, columnspan=3, pady=10)

root.mainloop()