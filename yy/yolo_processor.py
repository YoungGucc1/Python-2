import os
import cv2
import numpy as np
import torch # Keep for potential torch.hub or if Ultralytics uses it directly
from typing import Optional, List, Dict, Any

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not found. ONNX model loading disabled.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics package not found. Will try torch.hub if needed for .pt models.")


class YoloProcessor:
    """Handles loading and running YOLO models for object detection."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.model_type: Optional[str] = None  # 'ultralytics', 'torch', 'onnx'
        self.class_names: List[str] = [] 

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None and self.model_type is not None

    def load_model(self, model_path: Optional[str]) -> bool:
        """Load a YOLO model (.pt or .onnx). Handles model_path=None by clearing current model."""
        if not model_path: # Handles None or empty string
            self.model = None
            self.model_path = None
            self.model_type = None
            self.class_names = []
            # print("YoloProcessor: No model path provided, model state cleared.") # Optional debug
            return False # Indicate no model is loaded/active

        if not os.path.exists(model_path):
            print(f"Model Error: File not found at {model_path}")
            self.model = None; self.model_path = None; self.model_type = None; self.class_names = []
            return False

        # If loading a new model, reset previous state
        self.model = None
        self.model_path = None # Will be set on successful load
        self.model_type = None
        self.class_names = []
        
        current_model_path_to_load = model_path # Use a local variable for the path being loaded
        extension = os.path.splitext(current_model_path_to_load)[1].lower()

        try:
            if extension == '.pt':
                loaded_pt = False
                if ULTRALYTICS_AVAILABLE:
                    try:
                        temp_model = YOLO(current_model_path_to_load)
                        if hasattr(temp_model, 'names') and temp_model.names:
                             self.model = temp_model
                             self.model_type = 'ultralytics'
                             self.class_names = list(self.model.names.values())
                             print(f"Model Info: Loaded ultralytics YOLO model: {current_model_path_to_load}")
                             print(f"Model Info: Found {len(self.class_names)} classes: {self.class_names}")
                             loaded_pt = True
                        else:
                            print("Ultralytics model loaded but seems invalid (no names or empty names).")
                    except Exception as ultra_e:
                        print(f"Model Warning: Failed to load .pt with ultralytics ({ultra_e}). Trying torch.hub...")
                
                if not loaded_pt: # Try torch.hub if ultralytics failed or not available
                    try:
                        # Ensure torch is available for this path
                        # import torch # Already imported at top level
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=current_model_path_to_load, force_reload=False)
                        if hasattr(self.model, 'names') and self.model.names: # YOLOv5 models have .names list
                            self.model_type = 'torch'
                            # ultralytics/yolov5 model.names is often a list, not a dict
                            self.class_names = list(self.model.names) if isinstance(self.model.names, (list, dict)) else []
                            print(f"Model Info: Loaded torch.hub YOLOv5 model: {current_model_path_to_load}")
                            print(f"Model Info: Found {len(self.class_names)} classes: {self.class_names}")
                            loaded_pt = True
                        else:
                             print("Torch Hub model loaded but seems invalid (no names or empty names).")
                    except Exception as torch_e:
                        print(f"Model Error: Failed to load .pt model with torch.hub: {torch_e}")
                
                if loaded_pt:
                    self.model_path = current_model_path_to_load
                    return True
                return False # .pt loading failed through all avenues

            elif extension == '.onnx':
                if ONNX_AVAILABLE and ort:
                    try:
                        self.model = ort.InferenceSession(current_model_path_to_load, providers=['CPUExecutionProvider'])
                        self.model_type = 'onnx'
                        self.model_path = current_model_path_to_load
                        print(f"Model Info: Loaded ONNX model: {current_model_path_to_load}")
                        print("Model Info: Class names for ONNX models must be managed manually or via associated files.")
                        # self._try_get_onnx_metadata() # Consider future implementation
                        return True
                    except Exception as onnx_e:
                        print(f"Model Error: Failed to load ONNX model: {onnx_e}")
                        return False
                else:
                    print("Model Error: ONNX Runtime is not installed/available. Cannot load .onnx model.")
                    return False
            else:
                print(f"Model Error: Unsupported model format: {extension}")
                return False

        except Exception as e:
            print(f"Model Error: Unexpected error loading model {current_model_path_to_load}: {e}")
            self.model = None; self.model_path = None; self.model_type = None; self.class_names = []
            return False

    def detect(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Optional[List[Dict]]:
        """
        Run object detection on an image.
        Returns list of {'bbox_xyxy': [x1,y1,x2,y2], 'confidence': float, 'model_class_id': int} or None.
        """
        if not self.is_model_loaded():
            print("Detection Error: No model loaded.")
            return None

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Detection Error: Could not read image: {image_path}")
                return None

            detections = []

            if self.model_type == 'ultralytics':
                results = self.model(img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False) 
                if results:
                    result = results[0] 
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = boxes.conf[i].item()
                        cls_id = int(boxes.cls[i].item())
                        detections.append({
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'confidence': conf,
                            'model_class_id': cls_id
                        })

            elif self.model_type == 'torch': 
                results = self.model(img_bgr) 
                preds = results.xyxy[0].cpu().numpy() 
                for *xyxy, conf, cls_id in preds:
                    if conf >= conf_threshold: 
                         detections.append({
                            'bbox_xyxy': xyxy,
                            'confidence': conf,
                            'model_class_id': int(cls_id)
                         })

            elif self.model_type == 'onnx':
                detections = self._detect_onnx(img_bgr, conf_threshold, iou_threshold)

            else:
                print(f"Detection Error: Unsupported model type '{self.model_type}' for detection.")
                return None

            return detections

        except Exception as e:
            print(f"Detection Error: An error occurred during detection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_onnx(self, img_bgr: np.ndarray, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        """Internal helper for ONNX inference (example for YOLOv8 ONNX)."""
        if not ONNX_AVAILABLE or self.model is None: return []

        try:
            input_details = self.model.get_inputs()[0]
            input_name = input_details.name
            input_shape = input_details.shape 
            _, _, input_height, input_width = input_shape

            img_height_orig, img_width_orig = img_bgr.shape[:2]

            img_resized = cv2.resize(img_bgr, (input_width, input_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1)) 
            input_tensor = np.expand_dims(img_transposed, axis=0) 

            outputs = self.model.run(None, {input_name: input_tensor})
            
            # Assuming output shape [1, num_classes + 4, num_detections_proposal] e.g. [1, 84, 8400] for COCO
            # Or it could be [1, num_detections_proposal, num_classes + 4]
            # For many YOLOv8 ONNX exports, it's [1, 84, N]
            predictions_raw = np.squeeze(outputs[0]) # Becomes [84, N]
            if predictions_raw.shape[0] > predictions_raw.shape[1] and predictions_raw.shape[1] > (len(self.class_names) if self.class_names else 0): # Heuristic for [N, 84]
                 predictions = predictions_raw # If already [N, 84]
            else: # Assume [84, N]
                 predictions = predictions_raw.T # Transpose to [N, 84]

            boxes_data = predictions[:, :4] # cx, cy, w, h (normalized to input_width, input_height)
            scores_data = predictions[:, 4:] # Class scores

            # Filter by confidence score
            class_ids = np.argmax(scores_data, axis=1)
            confidences = np.max(scores_data, axis=1)
            
            mask = confidences > conf_threshold
            boxes_filtered = boxes_data[mask]
            confidences_filtered = confidences[mask]
            class_ids_filtered = class_ids[mask]

            if len(boxes_filtered) == 0:
                return []

            # Denormalize boxes to original image dimensions
            # Boxes are cx,cy,w,h relative to input_size (e.g. 640x640)
            # Convert to x1,y1,x2,y2 relative to original image size
            x_factor = img_width_orig / input_width
            y_factor = img_height_orig / input_height

            cx = boxes_filtered[:, 0] * x_factor
            cy = boxes_filtered[:, 1] * y_factor
            w = boxes_filtered[:, 2] * x_factor
            h = boxes_filtered[:, 3] * y_factor

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # OpenCV NMSBoxes expects (x, y, w, h) for boxes, not (x1,y1,x2,y2)
            # Alternatively, implement NMS that takes xyxy or use torchvision.ops.nms
            # For simplicity with cv2.dnn.NMSBoxes, let's use (x,y,w,h) format for NMS input
            # Note: x,y should be top-left corner for NMSBoxes
            nms_boxes = np.column_stack((x1, y1, w, h)).tolist()
            nms_confidences = confidences_filtered.tolist()
            
            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, conf_threshold, iou_threshold)

            final_detections = []
            if len(indices) > 0:
                 # NMSBoxes returns a 2D array if not empty, e.g., [[0], [2]]. Flatten if needed.
                 # Or iterate directly if it's a flat list/tuple of indices.
                 # cv2.dnn.NMSBoxes typically returns a column vector (2D) or empty tuple
                for i_arr in indices:
                    i = i_arr[0] # Get the actual index
                    final_detections.append({
                        'bbox_xyxy': [x1[i], y1[i], x2[i], y2[i]], # Use original xyxy calculated before NMS input conversion
                        'confidence': confidences_filtered[i].item(),
                        'model_class_id': class_ids_filtered[i].item()
                    })
            return final_detections

        except Exception as e:
            print(f"ONNX Detection Error: {e}")
            import traceback
            traceback.print_exc()
            return []
