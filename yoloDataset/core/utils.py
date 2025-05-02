"""
Utility functions for YOLO Dataset Creator
"""

import numpy as np
import cv2

def xyxy_to_cxcywh_normalized(xyxy, img_width, img_height):
    """
    Convert bounding box from [x1, y1, x2, y2] format (pixel coordinates) to 
    [center_x, center_y, width, height] format normalized (0-1)
    
    Args:
        xyxy (list or np.ndarray): Bounding box in [x1, y1, x2, y2] format (pixel coordinates)
        img_width (int): Width of the image
        img_height (int): Height of the image
        
    Returns:
        list: Bounding box in [center_x, center_y, width, height] format normalized (0-1)
    """
    x1, y1, x2, y2 = xyxy
    
    # Calculate width and height of the box
    width = x2 - x1
    height = y2 - y1
    
    # Calculate center coordinates
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    
    # Normalize
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [center_x_norm, center_y_norm, width_norm, height_norm]

def cxcywh_normalized_to_xyxy(cxcywh_norm, img_width, img_height):
    """
    Convert bounding box from [center_x, center_y, width, height] format normalized (0-1) to
    [x1, y1, x2, y2] format (pixel coordinates)
    
    Args:
        cxcywh_norm (list or np.ndarray): Bounding box in [center_x, center_y, width, height] format normalized (0-1)
        img_width (int): Width of the image
        img_height (int): Height of the image
        
    Returns:
        list: Bounding box in [x1, y1, x2, y2] format (pixel coordinates)
    """
    center_x_norm, center_y_norm, width_norm, height_norm = cxcywh_norm
    
    # Denormalize
    center_x = center_x_norm * img_width
    center_y = center_y_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height
    
    # Calculate corner coordinates
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    
    return [int(x1), int(y1), int(x2), int(y2)]

def cxcywh_normalized_to_xyxy_normalized(cxcywh_norm):
    """
    Convert bounding box from [center_x, center_y, width, height] format normalized (0-1) to
    [x1, y1, x2, y2] format normalized (0-1)
    
    Args:
        cxcywh_norm (list or np.ndarray): Bounding box in [center_x, center_y, width, height] format normalized (0-1)
        
    Returns:
        list: Bounding box in [x1, y1, x2, y2] format normalized (0-1)
    """
    center_x_norm, center_y_norm, width_norm, height_norm = cxcywh_norm
    
    # Calculate corner coordinates (still normalized)
    x1_norm = center_x_norm - width_norm / 2
    y1_norm = center_y_norm - height_norm / 2
    x2_norm = center_x_norm + width_norm / 2
    y2_norm = center_y_norm + height_norm / 2
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """
    Resize an image maintaining aspect ratio
    
    Args:
        image (np.ndarray): Image to resize
        target_width (int, optional): Target width. Defaults to None.
        target_height (int, optional): Target height. Defaults to None.
        
    Returns:
        np.ndarray: Resized image
    """
    h, w = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        aspect_ratio = target_height / h
        new_width = int(w * aspect_ratio)
        new_height = target_height
    elif target_height is None:
        aspect_ratio = target_width / w
        new_width = target_width
        new_height = int(h * aspect_ratio)
    else:
        # If both dimensions are provided, maintain aspect ratio based on the smaller dimension
        aspect_ratio_width = target_width / w
        aspect_ratio_height = target_height / h
        aspect_ratio = min(aspect_ratio_width, aspect_ratio_height)
        new_width = int(w * aspect_ratio)
        new_height = int(h * aspect_ratio)
        
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def ensure_dir(directory):
    """
    Make sure the directory exists, create it if it doesn't
    
    Args:
        directory (str): Directory path
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory) 