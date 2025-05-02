"""
ImageCanvas module - Custom widget for displaying images and bounding boxes
"""

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QFont, QFontMetrics, QImage
from PySide6.QtCore import Qt, QRect, Signal

import cv2
import numpy as np

from core.utils import cxcywh_normalized_to_xyxy

class ImageCanvas(QWidget):
    """
    Custom widget for displaying images and bounding boxes
    """
    boxSelected = Signal(int)  # Signal emitted when a box is selected (sends box index)
    
    def __init__(self, parent=None):
        """Initialize the ImageCanvas"""
        super().__init__(parent)
        
        # Display properties
        self.pixmap = None
        self.image_cv2 = None  # Original OpenCV image
        self.image_path = None
        self.aspect_ratio = 1.0
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Bounding box properties
        self.annotations = []  # List of annotations for the current image
        self.selected_box_idx = -1  # -1 means no box selected
        self.class_names = []  # Class names list
        self.default_class_colors = [
            QColor(255, 0, 0),     # Red
            QColor(0, 255, 0),     # Green
            QColor(0, 0, 255),     # Blue
            QColor(255, 255, 0),   # Yellow
            QColor(255, 0, 255),   # Magenta
            QColor(0, 255, 255),   # Cyan
            QColor(255, 128, 0),   # Orange
            QColor(128, 0, 255),   # Purple
            QColor(0, 128, 128),   # Teal
            QColor(128, 128, 0),   # Olive
        ]
        
        # Appearance
        self.box_line_width = 2
        self.label_font = QFont("Arial", 10)
        self.font_metrics = QFontMetrics(self.label_font)
        
        # Widget settings
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_image(self, image_path=None, cv2_image=None):
        """
        Load and display an image from file or from CV2 image object
        
        Args:
            image_path (str, optional): Path to the image file. Defaults to None.
            cv2_image (ndarray, optional): OpenCV image. Defaults to None.
            
        Returns:
            bool: True if image was loaded successfully, False otherwise
        """
        self.image_path = image_path
        self.selected_box_idx = -1  # Reset box selection
        
        if cv2_image is not None:
            self.image_cv2 = cv2_image
        elif image_path:
            # Load the image
            self.image_cv2 = cv2.imread(image_path)
            if self.image_cv2 is None:
                print(f"Failed to load image: {image_path}")
                self.pixmap = None
                self.update()
                return False
        else:
            self.image_cv2 = None
            self.pixmap = None
            self.update()
            return False
            
        # Convert to Qt format
        rgb_image = cv2.cvtColor(self.image_cv2, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Create QImage first, then convert to QPixmap
        qimg = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        
        # Store aspect ratio for proper display
        self.aspect_ratio = w / h if h > 0 else 1.0
        
        # Reset view transforms
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.update()
        return True
    
    def set_annotations(self, annotations):
        """
        Set the annotations for the current image
        
        Args:
            annotations (list): List of annotations in format 
                              [{'class_id': int, 'bbox': [cx, cy, w, h]}, ...]
        """
        self.annotations = annotations
        self.update()
    
    def set_class_names(self, class_names):
        """
        Set the class names list
        
        Args:
            class_names (list): List of class names
        """
        self.class_names = class_names
        self.update()
    
    def fit_to_view(self):
        """Scale image to fit the widget size"""
        if self.pixmap is None:
            return
            
        # Reset to fit view
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update()
    
    def zoom_in(self):
        """Zoom in on the image"""
        if self.pixmap is None:
            return
            
        self.scale_factor *= 1.2
        self.update()
    
    def zoom_out(self):
        """Zoom out of the image"""
        if self.pixmap is None:
            return
            
        self.scale_factor *= 0.8
        self.update()
    
    def paintEvent(self, event):
        """
        Draw the image and bounding boxes
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(64, 64, 64))  # Dark gray background
        
        if self.pixmap is None:
            # Draw a message or just leave it with the background color
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image Loaded")
            return
        
        # Calculate display size and position to maintain aspect ratio
        display_rect = self._get_display_rect()
        
        # Draw the image
        painter.drawPixmap(display_rect, self.pixmap)
        
        # Draw all bounding boxes
        if self.image_cv2 is not None and self.annotations:
            img_h, img_w = self.image_cv2.shape[:2]
            
            for i, annotation in enumerate(self.annotations):
                # Get normalized bounding box in YOLO format [cx, cy, w, h]
                bbox = annotation.get('bbox', [0, 0, 0, 0])
                class_id = annotation.get('class_id', 0)
                
                # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                x1, y1, x2, y2 = cxcywh_normalized_to_xyxy(bbox, img_w, img_h)
                
                # Scale to display coordinates
                box_x1, box_y1 = self._image_to_display_coords(x1, y1, display_rect)
                box_x2, box_y2 = self._image_to_display_coords(x2, y2, display_rect)
                
                # Draw the box
                box_color = self._get_color_for_class(class_id)
                if i == self.selected_box_idx:
                    # Highlight selected box
                    pen = QPen(box_color, self.box_line_width * 2)
                    pen.setStyle(Qt.DashLine)
                else:
                    pen = QPen(box_color, self.box_line_width)
                
                painter.setPen(pen)
                painter.drawRect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1)
                
                # Draw label
                class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"Class {class_id}"
                confidence = annotation.get('confidence', None)
                
                if confidence is not None:
                    label_text = f"{class_name} ({confidence:.2f})"
                else:
                    label_text = class_name
                
                # Calculate label position
                label_width = self.font_metrics.horizontalAdvance(label_text) + 10
                label_height = self.font_metrics.height() + 4
                
                # Place label at top of box if there's room, otherwise inside box at top
                label_x = box_x1
                label_y = box_y1 - label_height if box_y1 > label_height else box_y1
                
                # Draw label background
                painter.fillRect(label_x, label_y, label_width, label_height, box_color)
                
                # Draw label text
                painter.setPen(Qt.white)
                painter.setFont(self.label_font)
                painter.drawText(
                    label_x + 5, 
                    label_y + label_height - 5, 
                    label_text
                )
    
    def mousePressEvent(self, event):
        """
        Handle mouse press events
        
        Args:
            event: Mouse event
        """
        if event.button() == Qt.LeftButton and self.pixmap and self.annotations:
            # Get display area
            display_rect = self._get_display_rect()
            
            # Convert mouse position to image coordinates
            mouse_pos = event.position()
            
            # Check if click is inside any bounding box
            clicked_box_idx = -1
            
            img_h, img_w = self.image_cv2.shape[:2]
            
            for i, annotation in enumerate(self.annotations):
                bbox = annotation.get('bbox', [0, 0, 0, 0])
                
                # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                x1, y1, x2, y2 = cxcywh_normalized_to_xyxy(bbox, img_w, img_h)
                
                # Scale to display coordinates
                box_x1, box_y1 = self._image_to_display_coords(x1, y1, display_rect)
                box_x2, box_y2 = self._image_to_display_coords(x2, y2, display_rect)
                
                # Check if click is inside this box
                if (box_x1 <= mouse_pos.x() <= box_x2 and 
                    box_y1 <= mouse_pos.y() <= box_y2):
                    clicked_box_idx = i
                    break
            
            # Update selected box
            if clicked_box_idx != self.selected_box_idx:
                self.selected_box_idx = clicked_box_idx
                self.boxSelected.emit(clicked_box_idx)
                self.update()
                
    def _get_display_rect(self):
        """
        Calculate the display rectangle for the image
        
        Returns:
            QRect: Rectangle for displaying the image
        """
        if self.pixmap is None:
            return QRect()
            
        # Get widget dimensions
        widget_width = self.width()
        widget_height = self.height()
        
        # Calculate scaled dimensions
        scaled_width = widget_width * self.scale_factor
        scaled_height = scaled_width / self.aspect_ratio
        
        # Adjust if too tall
        if scaled_height > widget_height:
            scaled_height = widget_height * self.scale_factor
            scaled_width = scaled_height * self.aspect_ratio
        
        # Calculate position (centered)
        x = (widget_width - scaled_width) / 2 + self.offset_x
        y = (widget_height - scaled_height) / 2 + self.offset_y
        
        return QRect(int(x), int(y), int(scaled_width), int(scaled_height))
    
    def _image_to_display_coords(self, x, y, display_rect):
        """
        Convert image coordinates to display coordinates
        
        Args:
            x, y: Pixel coordinates in the original image
            display_rect: Rectangle where the image is displayed
            
        Returns:
            tuple: (display_x, display_y)
        """
        if self.image_cv2 is None:
            return (0, 0)
            
        img_h, img_w = self.image_cv2.shape[:2]
        
        # Convert to 0-1 range within the image
        normalized_x = x / img_w
        normalized_y = y / img_h
        
        # Convert to display coordinates
        display_x = display_rect.x() + normalized_x * display_rect.width()
        display_y = display_rect.y() + normalized_y * display_rect.height()
        
        return int(display_x), int(display_y)
    
    def _get_color_for_class(self, class_id):
        """
        Get a color for a class
        
        Args:
            class_id (int): Class ID
            
        Returns:
            QColor: Color for the class
        """
        if 0 <= class_id < len(self.default_class_colors):
            return self.default_class_colors[class_id]
        else:
            # Generate a color for classes beyond our predefined list
            # Use a deterministic hash-like function based on class_id
            h = (class_id * 8751) % 360  # Hue (0-360)
            s = ((class_id * 125) % 50) + 50  # Saturation (50-100%)
            v = 90  # Value: Consistently bright (90%)
            
            # Convert HSV to RGB for QColor
            h_i = int(h / 60)
            f = h / 60 - h_i
            p = v * (1 - s / 100)
            q = v * (1 - f * s / 100)
            t = v * (1 - (1 - f) * s / 100)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
                
            return QColor(int(r * 255 / 100), int(g * 255 / 100), int(b * 255 / 100)) 