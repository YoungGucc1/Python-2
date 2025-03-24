from PIL import Image
import os
import numpy as np

def process_vision_info(messages):
    """
    Process vision information from messages for the Qwen2-VL model.
    
    Args:
        messages (list): List of message dictionaries containing text and image data
        
    Returns:
        tuple: (image_inputs, video_inputs) where:
            - image_inputs is a list of PIL Images or None
            - video_inputs is a list of video data or None (usually empty for image-only tasks)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] != "user":
            continue
            
        for item in message["content"]:
            if item["type"] == "image":
                # If the image is a path, load it
                if isinstance(item["image"], str):
                    if os.path.exists(item["image"]):
                        try:
                            img = Image.open(item["image"])
                            image_inputs.append(img)
                        except Exception as e:
                            print(f"Error loading image from path: {e}")
                # If the image is already a PIL Image
                elif isinstance(item["image"], Image.Image):
                    image_inputs.append(item["image"])
                # If the image is a numpy array
                elif isinstance(item["image"], np.ndarray):
                    img = Image.fromarray(item["image"])
                    image_inputs.append(img)
                    
            elif item["type"] == "video":
                # Handle video inputs if needed
                video_inputs.append(item["video"])
    
    return image_inputs, video_inputs 