"""
Image Augmenter module - Handles image augmentation using Albumentations
"""

import cv2
import numpy as np
import albumentations as A

class ImageAugmenter:
    """
    Handles image augmentation using Albumentations library
    """
    
    def __init__(self):
        """Initialize the ImageAugmenter with default augmentation pipeline"""
        # Create a default augmentation pipeline
        self.transform = self._get_default_transform()
    
    def _get_default_transform(self):
        """
        Get a default augmentation transform pipeline
        
        Returns:
            A.Compose: Default augmentation pipeline
        """
        # Create a default augmentation pipeline that works well for object detection
        transform = A.Compose([
            # Spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            # Using Affine instead of ShiftScaleRotate as recommended
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
            
            # Color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            
            # Noise and blur augmentations
            A.GaussianBlur(blur_limit=3, p=0.2),
            # Updated GaussNoise parameters
            A.GaussNoise(std_range=(0.07, 0.15), mean_range=(0, 0), per_channel=True, p=0.2),
            
            # Weather augmentations - use proper parameters
            A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.08, p=0.1),
        ], 
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height] normalized
            label_fields=['class_labels']
        ))
        
        return transform
    
    def set_custom_transform(self, transform):
        """
        Set a custom augmentation pipeline
        
        Args:
            transform (A.Compose): Custom augmentation pipeline
        """
        self.transform = transform
    
    def augment_image(self, image, annotations, num_augmentations=5):
        """
        Apply augmentation to an image and its annotations
        
        Args:
            image (np.ndarray): Input image (BGR format, as returned by cv2.imread)
            annotations (list): List of annotations in format 
                              [{'class_id': int, 'bbox': [cx, cy, w, h]}, ...]
                              where bbox coordinates are normalized (0-1)
            num_augmentations (int): Number of augmented versions to generate
            
        Returns:
            list: List of tuples [(augmented_image, augmented_annotations), ...]
        """
        if len(annotations) == 0:
            return []  # Skip augmentation if no annotations
        
        results = []
        
        for _ in range(num_augmentations):
            # Extract bounding boxes and class_ids from annotations
            bboxes = [ann['bbox'] for ann in annotations]
            class_labels = [ann['class_id'] for ann in annotations]
            
            if not bboxes:
                continue  # Skip if no boxes
                
            # Apply augmentation
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Get augmented image and bounding boxes
                augmented_image = transformed['image']
                augmented_bboxes = transformed['bboxes']
                augmented_class_labels = transformed['class_labels']
                
                # Skip if no valid boxes after augmentation
                if not augmented_bboxes:
                    continue
                    
                # Reconstruct annotations in the original format
                augmented_annotations = []
                for i, bbox in enumerate(augmented_bboxes):
                    # Copy original annotation for confidence values if available
                    ann = {'class_id': augmented_class_labels[i], 'bbox': bbox}
                    if i < len(annotations) and 'confidence' in annotations[i]:
                        ann['confidence'] = annotations[i]['confidence']
                    augmented_annotations.append(ann)
                
                results.append((augmented_image, augmented_annotations))
                
            except Exception as e:
                print(f"Error during augmentation: {e}")
                continue
                
        return results
        
    def preview_augmentations(self, image, annotations, num_samples=3):
        """
        Generate preview of augmentations
        
        Args:
            image (np.ndarray): Input image
            annotations (list): List of annotations
            num_samples (int): Number of preview samples to generate
            
        Returns:
            list: List of tuples [(augmented_image, augmented_annotations), ...]
        """
        return self.augment_image(image, annotations, num_augmentations=num_samples) 