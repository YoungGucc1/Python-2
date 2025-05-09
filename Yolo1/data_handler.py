import os
import shutil
import random
import cv2
import yaml
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
from PyQt6.QtCore import QRect
from pathlib import Path

# Use local imports due to flat structure
from utils import ensure_dir, xyxy_to_cxcywh_normalized, cxcywh_normalized_to_xyxy, clamp

class DataHandler:
    """
    Handles all data management for the YOLO Dataset Creator application.
    Annotations are stored internally in YOLO format (normalized cx, cy, w, h).
    """
    def __init__(self):
        self.image_paths: List[str] = []
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []
        # {image_path: [{'class_id': int, 'bbox': [cx, cy, w, h], 'confidence': float, 'model_class_id': int}, ...]}
        self.annotations: Dict[str, List[dict]] = {}
        # {original_path: [(augmented_image_np, augmented_annotations_list), ...]}
        self.augmented_images: Dict[str, List[Tuple[np.ndarray, List[dict]]]] = {}
        self._image_dims_cache: Dict[str, Tuple[int, int]] = {} # Cache image dimensions (w, h)
        self._dirty: bool = False # Track unsaved changes

    # --- Dirty State Management ---
    def set_dirty(self, dirty: bool = True):
        self._dirty = dirty

    def is_dirty(self) -> bool:
        return self._dirty

    # --- Helper to get relative path ---
    def _get_relative_path(self, target_path: Optional[str], base_dir: Path) -> Optional[str]:
        if target_path is None:
            return None
        try:
            # Try to make path relative
            relative_path = Path(os.path.relpath(target_path, base_dir))
            return str(relative_path)
        except ValueError:
            # If paths are on different drives (Windows), keep absolute path
            return target_path

    # --- Helper to get absolute path ---
    def _get_absolute_path(self, relative_or_absolute_path: Optional[str], base_dir: Path) -> Optional[str]:
        if relative_or_absolute_path is None:
            return None
        path_obj = Path(relative_or_absolute_path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            # Resolve relative path based on the project file's directory
            absolute_path = (base_dir / path_obj).resolve()
            return str(absolute_path)


    # --- Project Save/Load ---
    def save_project(self, file_path: str) -> bool:
        """Save the current project state to a JSON file."""
        project_file = Path(file_path)
        base_dir = project_file.parent

        # Prepare data for saving (using relative paths where possible)
        project_data = {
            "version": "0.1.0", # Project file format version
            "model_path": self._get_relative_path(self.model_path, base_dir),
            "class_names": self.class_names,
            "image_data": {}
        }

        for img_path_abs in self.image_paths:
             img_path_rel = self._get_relative_path(img_path_abs, base_dir)
             if img_path_rel: # Only save if relative path could be determined
                 project_data["image_data"][img_path_rel] = {
                      "annotations": self.annotations.get(img_path_abs, [])
                 }

        try:
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4)
            self.set_dirty(False) 
            print(f"Project saved successfully to {file_path}")
            return True
        except IOError as e:
            print(f"Error saving project file {file_path}: {e}")
            return False
        except TypeError as e:
            print(f"Error serializing project data to JSON: {e}")
            return False

    def load_project(self, file_path: str) -> bool:
        """Load project state from a JSON file."""
        project_file = Path(file_path)
        if not project_file.is_file():
            print(f"Error: Project file not found: {file_path}")
            return False

        base_dir = project_file.parent

        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
        except IOError as e:
            print(f"Error reading project file {file_path}: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing project file {file_path}: {e}")
            return False

        self.image_paths = []
        self.model_path = None
        self.class_names = []
        self.annotations = {}
        self.augmented_images = {}
        self._image_dims_cache = {}
        self._dirty = False

        file_version = project_data.get("version", "0.0.0")

        self.class_names = project_data.get("class_names", [])
        relative_model_path = project_data.get("model_path")
        self.model_path = self._get_absolute_path(relative_model_path, base_dir)

        image_data = project_data.get("image_data", {})
        loaded_image_paths = []
        for img_path_rel, data in image_data.items():
            img_path_abs = self._get_absolute_path(img_path_rel, base_dir)
            if img_path_abs and os.path.isfile(img_path_abs): 
                 if self._get_image_dims(img_path_abs) != (0, 0): 
                    loaded_image_paths.append(img_path_abs)
                    self.annotations[img_path_abs] = data.get("annotations", [])
                 else:
                      print(f"Warning: Could not read dimensions for image {img_path_abs} referenced in project. Skipping.")
            else:
                 print(f"Warning: Image file not found: {img_path_abs} (relative: {img_path_rel}). Removing from project.")

        self.image_paths = loaded_image_paths
        self.set_dirty(False) 
        print(f"Project loaded successfully from {file_path}")
        return True

    def _get_image_dims(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions (width, height), using cache or loading."""
        if image_path in self._image_dims_cache:
            return self._image_dims_cache[image_path]
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                self._image_dims_cache[image_path] = (w, h)
                return w, h
        except Exception as e:
            print(f"Error reading image dimensions for {image_path}: {e}")
        return 0, 0 

    def add_images(self, file_paths: List[str]) -> int:
        """Add valid image files to the project."""
        added_count = 0
        current_paths = set(self.image_paths)
        changed = False 
        for path in file_paths:
            if path not in current_paths and os.path.isfile(path):
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    if self._get_image_dims(path) != (0, 0):
                        self.image_paths.append(path)
                        self.annotations[path] = [] 
                        added_count += 1
                        changed = True
                    else:
                         print(f"Warning: Could not read dimensions for {path}. Skipping.")
        if changed: self.set_dirty()
        return added_count

    def remove_image(self, image_path: str) -> bool:
        """Remove an image and its data."""
        if image_path in self.image_paths:
            self.image_paths.remove(image_path)
            self.annotations.pop(image_path, None)
            self.augmented_images.pop(image_path, None)
            self._image_dims_cache.pop(image_path, None)
            self.set_dirty()
            return True
        return False

    def set_model_path(self, model_path: Optional[str]) -> bool:
        """Set the path to the YOLO model. Returns True if path is valid or None, False otherwise."""
        new_path_to_set = None
        path_is_valid_or_none = False

        if model_path is None:
            new_path_to_set = None
            path_is_valid_or_none = True
        elif os.path.isfile(model_path):
            new_path_to_set = model_path
            path_is_valid_or_none = True
        else: # model_path is a non-None string, but not a file
            path_is_valid_or_none = False
            # Do not change self.model_path if the new one is invalid

        if path_is_valid_or_none:
            if new_path_to_set != self.model_path:
                self.model_path = new_path_to_set
                self.set_dirty()
            return True # Path was valid (file) or None (clearing)
        
        return False # Path was non-None and invalid


    def add_class(self, class_name: str) -> bool:
        """Add a new class name if it doesn't exist."""
        if class_name and class_name not in self.class_names:
            self.class_names.append(class_name)
            self.set_dirty()
            return True
        return False

    def remove_class(self, class_name: str) -> bool:
        """Remove a class and update annotations."""
        if class_name in self.class_names:
            try:
                class_id_to_remove = self.class_names.index(class_name)
                self.class_names.pop(class_id_to_remove) 

                for img_path in self.annotations:
                    updated_anns = []
                    for ann in self.annotations[img_path]:
                        current_id = ann['class_id']
                        if current_id == class_id_to_remove:
                            ann['class_id'] = -1 
                            updated_anns.append(ann)
                        elif current_id > class_id_to_remove:
                            ann['class_id'] -= 1 
                            updated_anns.append(ann)
                        else:
                            updated_anns.append(ann) 
                    self.annotations[img_path] = updated_anns
                for orig_path in self.augmented_images:
                     new_aug_list = []
                     for img_data, aug_anns in self.augmented_images[orig_path]:
                          updated_aug_anns = []
                          for ann in aug_anns:
                               current_id = ann['class_id']
                               if current_id == class_id_to_remove:
                                    ann['class_id'] = -1
                                    updated_aug_anns.append(ann)
                               elif current_id > class_id_to_remove:
                                    ann['class_id'] -= 1
                                    updated_aug_anns.append(ann)
                               else:
                                    updated_aug_anns.append(ann)
                          new_aug_list.append( (img_data, updated_aug_anns) )
                     self.augmented_images[orig_path] = new_aug_list

                self.set_dirty() 
                return True
            except ValueError: 
                return False
        return False

    def update_annotations_from_detections(self, image_path: str, detections: List[dict]):
        """
        Update annotations from model detections. Assigns class_id = -1.
        """
        if image_path not in self.image_paths:
            print(f"Warning: Cannot update annotations for unknown image path: {image_path}")
            return False

        img_w, img_h = self._get_image_dims(image_path)
        if img_w == 0 or img_h == 0:
            print(f"Warning: Cannot update annotations, invalid dimensions for: {image_path}")
            return False

        new_annotations = []
        for det in detections:
            xyxy = det['bbox_xyxy']
            yolo_bbox = xyxy_to_cxcywh_normalized(xyxy, img_w, img_h)
            yolo_bbox = [clamp(c, 0.0, 1.0) for c in yolo_bbox]

            annotation = {
                'class_id': -1, 
                'bbox': yolo_bbox,
                'confidence': det.get('confidence', 1.0),
                'model_class_id': det.get('model_class_id', -1) 
            }
            new_annotations.append(annotation)

        if self.annotations.get(image_path) != new_annotations:
             self.annotations[image_path] = new_annotations
             self.set_dirty()
        return True

    def add_manual_box(self, image_path: str, box_rect_img_coords: QRect) -> int:
        """
        Adds a manually drawn box. Returns index or -1 on failure.
        """
        if image_path not in self.image_paths: return -1

        img_w, img_h = self._get_image_dims(image_path)
        if img_w == 0 or img_h == 0: return -1

        xyxy = [
            box_rect_img_coords.left(), box_rect_img_coords.top(),
            box_rect_img_coords.right(), box_rect_img_coords.bottom()
        ]
        yolo_bbox = xyxy_to_cxcywh_normalized(xyxy, img_w, img_h)
        yolo_bbox = [clamp(c, 0.0, 1.0) for c in yolo_bbox] 

        annotation = {
            'class_id': -1, 
            'bbox': yolo_bbox,
            'confidence': 1.0, 
            'model_class_id': -1
        }
        self.annotations[image_path].append(annotation)
        self.set_dirty()
        return len(self.annotations[image_path]) - 1


    def assign_class_to_box(self, image_path: str, box_index: int, class_id: int) -> bool:
        """Assign a class_id to a specific bounding box."""
        current_class_id = -2 
        if (image_path in self.annotations and
            0 <= box_index < len(self.annotations[image_path])):
            current_class_id = self.annotations[image_path][box_index].get('class_id', -1)

        if current_class_id != class_id: 
            if (0 <= class_id < len(self.class_names)): 
                 self.annotations[image_path][box_index]['class_id'] = class_id
                 self.set_dirty()
                 return True
            elif (class_id == -1): 
                 self.annotations[image_path][box_index]['class_id'] = -1
                 self.set_dirty()
                 return True

            print(f"Warning: Failed to assign invalid class {class_id} to box {box_index} for image {image_path}")
            return False
        else:
            return True


    def delete_box(self, image_path: str, box_index: int) -> bool:
         """Delete a specific bounding box annotation."""
         if (image_path in self.annotations and
             0 <= box_index < len(self.annotations[image_path])):
             del self.annotations[image_path][box_index]
             self.set_dirty()
             return True
         return False


    def add_augmented_images(self, image_path: str, augmented_data: List[Tuple[np.ndarray, List[dict]]]):
        """Store augmented images and their annotations."""
        if image_path in self.image_paths:
            if image_path not in self.augmented_images:
                self.augmented_images[image_path] = []
            self.augmented_images[image_path].extend(augmented_data)
            return True
        return False

    def clear_augmented_images(self):
        """Clear all previously generated augmented images."""
        self.augmented_images = {}

    def save_yolo_dataset(self, output_dir: str, train_split: float = 0.8,
                          progress_callback: Optional[Callable[[int, int, str], None]] = None) -> bool:
        """Save the dataset in YOLO format."""
        if not self.class_names:
            print("Error: Cannot save dataset without class names defined.")
            if progress_callback: progress_callback(0, 1, "Error: No classes defined")
            return False

        annotated_paths = {
            p for p, anns in self.annotations.items()
            if any(a['class_id'] >= 0 for a in anns)
        }
        augmented_annotated = {
             p for p, aug_list in self.augmented_images.items()
             if any(any(a['class_id'] >= 0 for a in anns) for _, anns in aug_list)
        }
        
        paths_to_save_orig = [p for p in self.image_paths if p in annotated_paths]
        augmented_to_save_data = {p: data for p, data in self.augmented_images.items() if p in augmented_annotated}


        if not paths_to_save_orig and not augmented_to_save_data:
             print("Error: No images with assigned annotations found to save.")
             if progress_callback: progress_callback(0, 1, "Error: No annotated images to save")
             return False

        try:
            dataset_dir = os.path.join(output_dir, 'yolo_dataset') 
            images_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            train_dir = os.path.join(images_dir, 'train')
            val_dir = os.path.join(images_dir, 'val')
            train_labels_dir = os.path.join(labels_dir, 'train')
            val_labels_dir = os.path.join(labels_dir, 'val')

            ensure_dir(dataset_dir); ensure_dir(train_dir); ensure_dir(val_dir)
            ensure_dir(train_labels_dir); ensure_dir(val_labels_dir)

            # Store tuples: (source_ident, dest_img_folder, dest_lbl_folder, annotations, base_name, name_modifier)
            # source_ident: str (path) for original, np.ndarray for augmented
            all_data_items: List[Tuple[Union[str, np.ndarray], Optional[str], Optional[str], List[Dict], str, str]] = []

            for img_path in paths_to_save_orig:
                 valid_anns = [ann for ann in self.annotations.get(img_path, []) if ann['class_id'] >= 0]
                 if valid_anns:
                      base_name = os.path.splitext(os.path.basename(img_path))[0]
                      all_data_items.append((img_path, None, None, valid_anns, base_name, ""))

            for orig_img_path, augmented_list in augmented_to_save_data.items():
                orig_base_name = os.path.splitext(os.path.basename(orig_img_path))[0]
                for j, (aug_img_np, aug_annotations) in enumerate(augmented_list):
                    valid_aug_anns = [ann for ann in aug_annotations if ann['class_id'] >= 0]
                    if valid_aug_anns:
                         modifier = f"_aug_{j+1}"
                         all_data_items.append((aug_img_np, None, None, valid_aug_anns, orig_base_name, modifier))


            total_items_to_process = len(all_data_items)
            if total_items_to_process == 0:
                 print("Error: No valid annotations found in selected images or augmentations.")
                 if progress_callback: progress_callback(0, 1, "Error: No valid annotations to save")
                 return False

            random.shuffle(all_data_items)
            split_idx = int(total_items_to_process * train_split)
            train_items_proto = all_data_items[:split_idx]
            val_items_proto = all_data_items[split_idx:]

            # Assign destination folders
            # Tuple: (source_ident, dest_img_folder, dest_lbl_folder, annotations, base_name, name_modifier)
            train_items = [(item[0], train_dir, train_labels_dir, item[3], item[4], item[5]) for item in train_items_proto]
            val_items = [(item[0], val_dir, val_labels_dir, item[3], item[4], item[5]) for item in val_items_proto]
            
            combined_items = train_items + val_items
            total_yaml_steps = total_items_to_process + 1 # +1 for data.yaml

            saved_image_paths_rel = {'train': [], 'val': []} 
            processed_count = 0

            for item_data in combined_items:
                processed_count += 1
                source_ident, dest_img_dir, dest_lbl_dir, annotations_to_save, base_name_stem, name_modifier_part = item_data
                set_name = os.path.basename(dest_img_dir) 

                if progress_callback:
                    progress_callback(processed_count, total_yaml_steps, f"Saving {set_name} item {processed_count}/{total_items_to_process}")

                img_to_save = None
                if isinstance(source_ident, str): 
                    img_to_save = cv2.imread(source_ident)
                    if img_to_save is None: 
                        print(f"Warning: Could not read image {source_ident} during save. Skipping.")
                        continue 
                elif isinstance(source_ident, np.ndarray): 
                    img_to_save = source_ident
                else: continue 

                final_base_name = f"{base_name_stem}{name_modifier_part}"
                dest_img_path = os.path.join(dest_img_dir, f"{final_base_name}.jpg")
                cv2.imwrite(dest_img_path, img_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                label_path = os.path.join(dest_lbl_dir, f"{final_base_name}.txt")
                with open(label_path, 'w') as f:
                    for ann in annotations_to_save:
                        class_id = ann['class_id']
                        bbox = ann['bbox'] 
                        bbox_str = ' '.join([f"{coord:.6f}" for coord in bbox])
                        f.write(f"{class_id} {bbox_str}\n")

                rel_img_path = os.path.join('images', set_name, f"{final_base_name}.jpg")
                saved_image_paths_rel[set_name].append(rel_img_path)


            if progress_callback:
                progress_callback(total_yaml_steps, total_yaml_steps, "Saving data.yaml")

            data_yaml_content = {
                 'path': dataset_dir, 
                 'train': os.path.join('images', 'train'), 
                 'val': os.path.join('images', 'val'),     
                 'nc': len(self.class_names),
                 'names': self.class_names
            }

            with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
                yaml.dump(data_yaml_content, f, default_flow_style=None, sort_keys=False)

            if progress_callback:
                progress_callback(total_yaml_steps, total_yaml_steps, "Dataset saved successfully")

            return True

        except Exception as e:
            print(f"Error saving dataset: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(processed_count if 'processed_count' in locals() else 0, 
                                  total_yaml_steps if 'total_yaml_steps' in locals() else 1, f"Error: {e}")
            return False