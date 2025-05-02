"""
DataHandler module - Manages the application's data including images, annotations, and classes
"""

import os
import shutil
import random
import cv2
import yaml
import numpy as np
from yolo_dataset_creator.core.utils import ensure_dir, xyxy_to_cxcywh_normalized

class DataHandler:
    """
    Handles all data management for the YOLO Dataset Creator application
    """
    
    def __init__(self):
        """Initialize the DataHandler"""
        self.image_paths = []  # List of paths to loaded images
        self.model_path = None  # Path to the selected YOLO model
        self.class_names = []  # List of class names
        # Dictionary of annotations: {image_path: [{'class_id': int, 'bbox': [cx, cy, w, h]}, ...]}
        self.annotations = {}
        # Dictionary of augmented images: {original_path: [(augmented_image, annotations), ...]}
        self.augmented_images = {}
    
    def add_images(self, file_paths):
        """
        Add image files to the project
        
        Args:
            file_paths (list): List of file paths to add
            
        Returns:
            int: Number of images successfully added
        """
        added_count = 0
        for path in file_paths:
            # Verify this is an image file
            try:
                img = cv2.imread(path)
                if img is not None:
                    # If this path isn't already in our list, add it and initialize empty annotations
                    if path not in self.image_paths:
                        self.image_paths.append(path)
                        self.annotations[path] = []
                        added_count += 1
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                
        return added_count
    
    def remove_image(self, image_path):
        """
        Remove an image and its annotations from the project
        
        Args:
            image_path (str): Path to the image to remove
            
        Returns:
            bool: True if the image was successfully removed, False otherwise
        """
        if image_path in self.image_paths:
            self.image_paths.remove(image_path)
            if image_path in self.annotations:
                del self.annotations[image_path]
            if image_path in self.augmented_images:
                del self.augmented_images[image_path]
            return True
        return False
    
    def set_model_path(self, model_path):
        """
        Set the path to the YOLO model
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if the model path was successfully set, False otherwise
        """
        if os.path.exists(model_path):
            self.model_path = model_path
            return True
        return False
    
    def add_class(self, class_name):
        """
        Add a class to the project
        
        Args:
            class_name (str): Name of the class to add
            
        Returns:
            bool: True if the class was successfully added, False otherwise
        """
        if class_name and class_name not in self.class_names:
            self.class_names.append(class_name)
            return True
        return False
    
    def remove_class(self, class_name):
        """
        Remove a class from the project and update any annotations that used this class
        
        Args:
            class_name (str): Name of the class to remove
            
        Returns:
            bool: True if the class was successfully removed, False otherwise
        """
        if class_name in self.class_names:
            class_id = self.class_names.index(class_name)
            self.class_names.remove(class_name)
            
            # Update annotations that used this class
            for img_path, annotations in self.annotations.items():
                # Remove boxes with this class
                self.annotations[img_path] = [ann for ann in annotations if ann['class_id'] != class_id]
                
                # Update class_ids for boxes with higher class_ids (they need to be decremented)
                for ann in self.annotations[img_path]:
                    if ann['class_id'] > class_id:
                        ann['class_id'] -= 1
            
            return True
        return False
    
    def update_annotations(self, image_path, bbox_detections, default_class=0):
        """
        Update annotations for an image from detection results
        
        Args:
            image_path (str): Path to the image
            bbox_detections (list): List of detected bounding boxes, each in format 
                                   {'bbox_xyxy': [x1, y1, x2, y2], 'confidence': float}
            default_class (int, optional): Default class ID to assign. Defaults to 0.
            
        Returns:
            bool: True if annotations were successfully updated
        """
        if image_path not in self.image_paths:
            return False
        
        # Clear any existing annotations for this image
        self.annotations[image_path] = []
        
        # Load the image to get dimensions for normalization
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        h, w = img.shape[:2]
        
        # Convert detections to our annotation format
        for det in bbox_detections:
            xyxy = det['bbox_xyxy']
            # Convert to YOLO format (normalized cx, cy, w, h)
            yolo_bbox = xyxy_to_cxcywh_normalized(xyxy, w, h)
            
            # Store annotation
            annotation = {
                'class_id': default_class,
                'bbox': yolo_bbox,
                'confidence': det.get('confidence', 1.0)
            }
            self.annotations[image_path].append(annotation)
            
        return True
    
    def assign_class_to_box(self, image_path, box_index, class_id):
        """
        Assign a class to a specific bounding box
        
        Args:
            image_path (str): Path to the image
            box_index (int): Index of the bounding box in the annotations list
            class_id (int): Class ID to assign
            
        Returns:
            bool: True if the class was successfully assigned, False otherwise
        """
        if (image_path in self.annotations and 
            0 <= box_index < len(self.annotations[image_path]) and 
            0 <= class_id < len(self.class_names)):
            
            self.annotations[image_path][box_index]['class_id'] = class_id
            return True
        
        return False
    
    def add_augmented_images(self, image_path, augmented_data):
        """
        Add augmented images and their annotations
        
        Args:
            image_path (str): Path to the original image
            augmented_data (list): List of tuples (augmented_image, annotations)
            
        Returns:
            bool: True if the augmented images were successfully added
        """
        if image_path in self.image_paths:
            self.augmented_images[image_path] = augmented_data
            return True
        return False
    
    def save_yolo_dataset(self, output_dir, train_split=0.8, progress_callback=None):
        """
        Save the dataset in YOLO format
        
        Args:
            output_dir (str): Base directory to save the dataset
            train_split (float, optional): Proportion of images to use for training. Defaults to 0.8.
            progress_callback (function, optional): Callback function to report progress.
                The function should accept three arguments: current step, total steps, and a status message.
            
        Returns:
            bool: True if the dataset was successfully saved, False otherwise
        """
        try:
            dataset_dir = os.path.join(output_dir, 'dataset')
            images_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            
            # Create directories
            ensure_dir(dataset_dir)
            ensure_dir(images_dir)
            ensure_dir(labels_dir)
            
            # If no images to save, return False
            if not self.image_paths and not any(self.augmented_images.values()):
                return False
                
            total_steps = (len(self.image_paths) + 
                          sum(len(aug_list) for aug_list in self.augmented_images.values()) +
                          3)  # +3 for train.txt, val.txt, and data.yaml
            current_step = 0
            
            # Prepare lists for train/val split
            all_image_files = []
            
            # Process original images
            for i, img_path in enumerate(self.image_paths):
                if progress_callback:
                    current_step += 1
                    progress_callback(current_step, total_steps, f"Processing image {i+1}/{len(self.image_paths)}")
                
                # Get filename without extension and full filename
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                dest_img_path = os.path.join(images_dir, f"{base_name}.jpg")
                
                # Copy and convert image to jpg
                img = cv2.imread(img_path)
                if img is None:
                    continue
                cv2.imwrite(dest_img_path, img)
                
                # Add to the list for train/val split
                all_image_files.append(dest_img_path)
                
                # Save corresponding label file
                self._save_yolo_label_file(labels_dir, base_name, self.annotations.get(img_path, []))
            
            # Process augmented images
            aug_counter = 0
            for orig_img_path, augmented_list in self.augmented_images.items():
                orig_base_name = os.path.splitext(os.path.basename(orig_img_path))[0]
                
                for j, (aug_img, aug_annotations) in enumerate(augmented_list):
                    if progress_callback:
                        current_step += 1
                        progress_callback(current_step, total_steps, 
                                        f"Processing augmented image {aug_counter+1}/{sum(len(l) for l in self.augmented_images.values())}")
                    
                    # Generate unique filename for augmented image
                    aug_name = f"{orig_base_name}_aug{j+1}.jpg"
                    dest_img_path = os.path.join(images_dir, aug_name)
                    
                    # Save augmented image
                    cv2.imwrite(dest_img_path, aug_img)
                    
                    # Add to the list for train/val split
                    all_image_files.append(dest_img_path)
                    
                    # Save corresponding label file
                    self._save_yolo_label_file(labels_dir, os.path.splitext(aug_name)[0], aug_annotations)
                    
                    aug_counter += 1
            
            # Create train/val split
            random.shuffle(all_image_files)
            split_idx = int(len(all_image_files) * train_split)
            train_files = all_image_files[:split_idx]
            val_files = all_image_files[split_idx:]
            
            # Convert paths to be relative to the dataset directory
            dataset_dir_rel = os.path.relpath(dataset_dir, output_dir)
            train_files_rel = [os.path.relpath(f, output_dir) for f in train_files]
            val_files_rel = [os.path.relpath(f, output_dir) for f in val_files]
            
            # Save train.txt and val.txt
            if progress_callback:
                current_step += 1
                progress_callback(current_step, total_steps, "Saving train.txt and val.txt")
                
            train_txt_path = os.path.join(dataset_dir, 'train.txt')
            val_txt_path = os.path.join(dataset_dir, 'val.txt')
            
            with open(train_txt_path, 'w') as f:
                f.write('\n'.join(train_files_rel))
                
            with open(val_txt_path, 'w') as f:
                f.write('\n'.join(val_files_rel))
            
            # Save data.yaml
            if progress_callback:
                current_step += 1
                progress_callback(current_step, total_steps, "Saving data.yaml")
                
            data_yaml = {
                'train': os.path.join(dataset_dir_rel, 'train.txt'),
                'val': os.path.join(dataset_dir_rel, 'val.txt'),
                'nc': len(self.class_names),
                'names': self.class_names
            }
            
            with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            if progress_callback:
                current_step += 1
                progress_callback(current_step, total_steps, "Dataset saved successfully")
                
            return True
            
        except Exception as e:
            print(f"Error saving dataset: {e}")
            if progress_callback:
                progress_callback(-1, total_steps, f"Error: {e}")
            return False
    
    def _save_yolo_label_file(self, labels_dir, base_name, annotations):
        """
        Save a YOLO format label file
        
        Args:
            labels_dir (str): Directory to save the label file
            base_name (str): Base name for the label file (without extension)
            annotations (list): List of annotations for this image
        """
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                bbox = ann['bbox']  # Already in YOLO format [cx, cy, w, h]
                bbox_str = ' '.join([f"{coord:.6f}" for coord in bbox])
                f.write(f"{class_id} {bbox_str}\n") 