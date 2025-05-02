"""
Image Augmenter module - Handles image augmentation using Albumentations (PyQt6 version)
"""

import cv2
import numpy as np
import albumentations as A
from typing import List, Tuple, Dict, Optional

class ImageAugmenter:
    """Handles image augmentation using Albumentations library."""

    def __init__(self):
        self.transform = self._get_default_transform()

    def _get_default_transform(self) -> A.Compose:
        """Get a default augmentation transform pipeline suitable for object detection."""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1), # Often less useful for common objects
            # A.RandomRotate90(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7,
                               border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]), # Gray border

            # Color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),

            # Blur/Noise
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # Adjust var_limit based on image intensity range

            # Quality/Compression related
            A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),

            # Geometric distortions
            A.GridDistortion(p=0.1),
            # A.OpticalDistortion(p=0.1), # Can be strong

            # Weather (optional, use carefully)
            # A.RandomRain(p=0.05)
            # A.RandomFog(p=0.05)

        ],
        bbox_params=A.BboxParams(
            format='yolo',  # [x_center, y_center, width, height] normalized
            label_fields=['class_labels'],
            min_visibility=0.2, # Discard boxes less than 20% visible
            min_area=16 # Discard boxes smaller than 4x4 pixels (adjust as needed)
        ))
        return transform

    def set_custom_transform(self, transform: A.Compose):
        """Set a custom Albumentations augmentation pipeline."""
        if not isinstance(transform, A.Compose):
            raise ValueError("Transform must be an instance of albumentations.Compose")
        if not transform.bbox_params:
             raise ValueError("Custom transform must include BboxParams with format='yolo' and label_fields=['class_labels']")
        if transform.bbox_params.format != 'yolo':
             raise ValueError("Custom transform BboxParams must use format='yolo'")
        if 'class_labels' not in transform.bbox_params.label_fields:
             raise ValueError("Custom transform BboxParams must have 'class_labels' in label_fields")

        self.transform = transform

    def augment_image(self,
                      image_bgr: np.ndarray,
                      annotations: List[Dict],
                      num_augmentations: int = 5
                      ) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        Apply augmentations to a BGR image and its annotations.

        Args:
            image_bgr: Input image (BGR format numpy array).
            annotations: List [{'class_id': int, 'bbox': [cx, cy, w, h]}, ...].
                         Only boxes with class_id >= 0 will be used for augmentation.
            num_augmentations: Number of augmented versions to generate.

        Returns:
            List of tuples [(augmented_image_bgr, augmented_annotations), ...].
            Returns empty list if no valid annotations are provided or if errors occur.
        """
        # Filter annotations to only include those with valid class IDs for augmentation
        valid_annotations = [ann for ann in annotations if ann.get('class_id', -1) >= 0]

        if not valid_annotations:
            # print("Augmentation skipped: No valid annotations provided.")
            return []

        results = []
        bboxes_orig = [ann['bbox'] for ann in valid_annotations]
        class_labels_orig = [ann['class_id'] for ann in valid_annotations]
        # Store other data like confidence if needed later
        other_data_orig = [{k: v for k, v in ann.items() if k not in ['bbox', 'class_id']} for ann in valid_annotations]


        for _ in range(num_augmentations):
            try:
                # Important: Convert BGR to RGB for Albumentations
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                transformed = self.transform(
                    image=image_rgb,
                    bboxes=bboxes_orig,
                    class_labels=class_labels_orig
                )

                augmented_image_rgb = transformed['image']
                augmented_bboxes = transformed['bboxes']
                augmented_class_labels = transformed['class_labels']

                # Skip if augmentation removed all boxes
                if not augmented_bboxes:
                    continue

                # Convert augmented image back to BGR for consistency
                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

                # Reconstruct annotations, clamping bbox coords
                augmented_annotations = []
                for i, bbox in enumerate(augmented_bboxes):
                    clamped_bbox = [clamp(c, 0.0, 1.0) for c in bbox] # Ensure bounds 0-1
                    # Check width/height again after clamping (optional)
                    if clamped_bbox[2] <= 0 or clamped_bbox[3] <= 0: continue

                    new_ann = {
                        'class_id': augmented_class_labels[i],
                        'bbox': clamped_bbox
                    }
                    # Add back other data if it existed (e.g., confidence, model_class_id)
                    # This assumes the order is preserved by albumentations
                    if i < len(other_data_orig):
                        new_ann.update(other_data_orig[i])

                    augmented_annotations.append(new_ann)

                # Only add if there are still valid annotations after filtering
                if augmented_annotations:
                    results.append((augmented_image_bgr, augmented_annotations))

            except Exception as e:
                print(f"Error during augmentation iteration: {e}")
                # import traceback
                # traceback.print_exc()
                continue # Skip this iteration on error

        return results

    def preview_augmentations(self,
                             image_bgr: np.ndarray,
                             annotations: List[Dict],
                             num_samples: int = 3
                             ) -> List[Tuple[np.ndarray, List[Dict]]]:
        """Generate a few samples of augmentations for preview."""
        return self.augment_image(image_bgr, annotations, num_augmentations=num_samples)