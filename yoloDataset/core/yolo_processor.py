"""
YoloProcessor module - Handles YOLO model loading and inference
"""

import os
import cv2
import numpy as np
import torch

class YoloProcessor:
    """
    Handles loading and running YOLO models for object detection
    """
    
    def __init__(self):
        """Initialize the YoloProcessor"""
        self.model = None
        self.model_path = None
        self.model_type = None  # 'torch' or 'onnx'
    
    def load_model(self, model_path):
        """
        Load a YOLO model
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if the model was successfully loaded, False otherwise
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        self.model_path = model_path
        extension = os.path.splitext(model_path)[1].lower()
        
        try:
            if extension == '.pt':
                # Load PyTorch model using ultralytics
                try:
                    # First try to import ultralytics package for most modern YOLO models
                    from ultralytics import YOLO
                    self.model = YOLO(model_path)
                    self.model_type = 'ultralytics'
                    print(f"Loaded ultralytics YOLO model: {model_path}")
                except ImportError:
                    # Fall back to torch hub if ultralytics isn't available
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                    self.model_type = 'torch'
                    print(f"Loaded torch hub YOLO model: {model_path}")
                
            elif extension == '.onnx':
                # Load ONNX model
                import onnxruntime as ort
                self.model = ort.InferenceSession(model_path)
                self.model_type = 'onnx'
                print(f"Loaded ONNX model: {model_path}")
            else:
                print(f"Unsupported model format: {extension}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_type = None
            return False
    
    def detect(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run object detection on an image
        
        Args:
            image_path (str): Path to the image file
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.45.
            
        Returns:
            list: List of detections, each in format 
                 {'bbox_xyxy': [x1, y1, x2, y2], 'confidence': float, 'model_class_id': int}
            None: If an error occurs
        """
        if self.model is None:
            print("No model loaded")
            return None
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None
                
            # Process based on model type
            if self.model_type == 'ultralytics':
                # Process with Ultralytics YOLO
                results = self.model(img, conf=conf_threshold, iou=iou_threshold)
                
                # Extract detections
                detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        
                        detections.append({
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'confidence': conf,
                            'model_class_id': cls_id
                        })
                
                return detections
                
            elif self.model_type == 'torch':
                # Process with PyTorch model (torch hub)
                results = self.model(img)
                
                # Extract predictions
                predictions = results.xyxy[0]  # Predictions (tensor): x1, y1, x2, y2, conf, class
                
                detections = []
                for pred in predictions:
                    x1, y1, x2, y2, conf, cls_id = pred.tolist()
                    
                    if conf >= conf_threshold:
                        detections.append({
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'confidence': conf,
                            'model_class_id': int(cls_id)
                        })
                        
                return detections
                
            elif self.model_type == 'onnx':
                # Process with ONNX model
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Preprocess image (resize, normalize, etc.) - this may need to be customized
                # based on the model requirements
                input_shape = self.model.get_inputs()[0].shape[1:3]  # Get input shape (height, width)
                if input_shape[0] != img_rgb.shape[0] or input_shape[1] != img_rgb.shape[1]:
                    img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[0]))
                else:
                    img_resized = img_rgb
                
                # Normalize pixel values
                img_input = img_resized.astype(np.float32) / 255.0
                
                # Add batch dimension and transpose to NCHW if needed
                img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
                img_input = np.expand_dims(img_input, 0)  # Add batch dimension
                
                # Get input and output names
                input_name = self.model.get_inputs()[0].name
                output_names = [output.name for output in self.model.get_outputs()]
                
                # Run inference
                outputs = self.model.run(output_names, {input_name: img_input})
                
                # Process outputs (format depends on the specific ONNX model)
                # This is a simplified example and may need to be adapted
                detections = []
                
                # Assuming standard YOLO output format
                if len(outputs) > 0:
                    # Get detection results (typically shape: [batch, num_detections, 5+num_classes])
                    results = outputs[0]
                    
                    # Process each detection
                    for detection in results[0]:
                        # First 4 elements are box coordinates, 5th is objectness score
                        # Remaining elements are class scores
                        if detection[4] > conf_threshold:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            
                            if confidence > conf_threshold:
                                # Adjust coordinates based on the original image size
                                x1, y1, x2, y2 = detection[0:4]
                                
                                # Convert to pixel coordinates on original image
                                orig_h, orig_w = img.shape[:2]
                                model_h, model_w = input_shape
                                
                                x1 = x1 / model_w * orig_w
                                y1 = y1 / model_h * orig_h
                                x2 = x2 / model_w * orig_w
                                y2 = y2 / model_h * orig_h
                                
                                detections.append({
                                    'bbox_xyxy': [x1, y1, x2, y2],
                                    'confidence': float(confidence),
                                    'model_class_id': int(class_id)
                                })
                
                return detections
            
            else:
                print(f"Unsupported model type: {self.model_type}")
                return None
                
        except Exception as e:
            print(f"Error during detection: {e}")
            return None 