#!/usr/bin/env python
"""
YOLO Dataset Creator - Launcher Script

Run this script to start the YOLO Dataset Creator application.
"""

import os
import sys

# Add the parent directory to the path so Python can find the yolo_dataset_creator package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if __name__ == "__main__":
    try:
        # Import and run the main function
        from yolo_dataset_creator.main import main
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed all the required dependencies:")
        print("pip install -r yolo_dataset_creator/requirements.txt") 