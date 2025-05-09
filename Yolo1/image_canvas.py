from PyQt6.QtWidgets import QWidget, QSizePolicy, QApplication
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen, QFont, QFontMetrics, QImage, QCursor
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Use local imports due to flat structure
from utils import cxcywh_normalized_to_xyxy, calculate_contrast_color, cv2_to_qimage, clamp

class ImageCanvas(QWidget):
    """
    Custom widget for displaying images, bounding boxes, and handling interactions.
    Supports zooming, panning, box selection, and drawing new boxes.
    """
    boxSelected = pyqtSignal(int)  # Signal emitted when a box is selected (sends box index, -1 for none)
    newBoxDrawn = pyqtSignal(QRect) # Signal emitted when a new box is drawn (pixel coords in image)
    requestContextMenu = pyqtSignal(QPoint) # Signal for context menu requests
    deleteSelectionKeyPressed = pyqtSignal() # Signal emitted when delete/backspace pressed on a selected box


    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Image Display
        self.pixmap: Optional[QPixmap] = None
        self.image_cv2_bgr: Optional[np.ndarray] = None 
        self.image_path: Optional[str] = None
        self.aspect_ratio: float = 1.0
        self.scale_factor: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.display_rect: QRect = QRect() 

        # Annotations
        self.annotations: List[dict] = []
        self.selected_box_idx: int = -1
        self.class_names: List[str] = []
        self.default_class_colors: List[QColor] = [
            QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0),
            QColor(255, 0, 255), QColor(0, 255, 255), QColor(255, 128, 0), QColor(128, 0, 255),
            QColor(0, 128, 128), QColor(128, 128, 0),
        ]
        self.unassigned_color = QColor(128, 128, 128, 180) 

        # Appearance
        self.box_line_width: int = 2
        self.label_font: QFont = QFont("Arial", 10)
        self.font_metrics: QFontMetrics = QFontMetrics(self.label_font)
        self.background_color: QColor = QColor(64, 64, 64)

        # Interaction State
        self._panning: bool = False
        self._drawing_box: bool = False
        self._is_drawing_enabled: bool = False
        self._start_drag_pos: Optional[QPoint] = None
        self._current_drag_pos: Optional[QPoint] = None
        self._last_mouse_pos: Optional[QPoint] = None

        # Widget Settings
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True) 
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) 

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update)
        self._update_timer.setInterval(50) 


    def set_image(self, image_path: Optional[str] = None, cv2_image: Optional[np.ndarray] = None) -> bool:
        """Load and display an image from path or CV2 BGR image object."""
        self.image_path = image_path
        self.selected_box_idx = -1
        self._reset_interactions()

        temp_image_cv2 = None
        if cv2_image is not None:
            temp_image_cv2 = cv2_image
        elif image_path:
            temp_image_cv2 = cv2.imread(image_path)
            if temp_image_cv2 is None:
                print(f"Failed to load image: {image_path}")
                self.pixmap = None
                self.image_cv2_bgr = None
                self.update()
                return False
        else:
            self.pixmap = None
            self.image_cv2_bgr = None
            self.update()
            return False

        self.image_cv2_bgr = temp_image_cv2 
        h, w = self.image_cv2_bgr.shape[:2]

        rgb_image = cv2.cvtColor(self.image_cv2_bgr, cv2.COLOR_BGR2RGB)
        qimg = cv2_to_qimage(rgb_image)

        if qimg is None:
            print("Failed to convert CV2 image to QImage.")
            self.pixmap = None
            self.image_cv2_bgr = None
            self.update()
            return False

        self.pixmap = QPixmap.fromImage(qimg)
        self.aspect_ratio = w / h if h > 0 else 1.0
        self.fit_to_view() 
        return True

    def set_annotations(self, annotations: List[dict]):
        """Set annotations. Expected: [{'class_id': int, 'bbox': [cx, cy, w, h]}, ...]"""
        self.annotations = annotations
        self.update()

    def set_class_names(self, class_names: List[str]):
        """Set the list of class names."""
        self.class_names = class_names
        self.update()

    def select_box(self, index: int):
        """Programmatically select a bounding box."""
        if 0 <= index < len(self.annotations):
            if self.selected_box_idx != index:
                self.selected_box_idx = index
                self.boxSelected.emit(index)
                self.update()
        elif self.selected_box_idx != -1:
             self.selected_box_idx = -1
             self.boxSelected.emit(-1)
             self.update()

    def set_drawing_enabled(self, enabled: bool):
        """Enable or disable drawing mode."""
        self._is_drawing_enabled = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.select_box(-1) 
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._reset_drawing() 

    # --- Zoom and Pan ---

    def fit_to_view(self):
        """Scale image to fit the widget size."""
        if self.pixmap is None: return
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._calculate_display_rect() 
        self.update()

    def zoom(self, factor: float, anchor_point: Optional[QPoint] = None):
        """Zoom in/out centered around an anchor point (widget coordinates)."""
        if self.pixmap is None: return

        if anchor_point is None:
            anchor_point = QPoint(self.width() // 2, self.height() // 2)

        img_x, img_y = self._display_to_image_coords(anchor_point.x(), anchor_point.y())
        
        # Limit zoom factor
        min_scale_factor = 0.1 
        max_scale_factor = 10.0
        current_pixmap_width_on_screen = self.pixmap.width() * self.scale_factor
        
        # Prevent zooming out too much if image is smaller than widget
        if factor < 1.0: # Zooming out
            widget_min_dim = min(self.width(), self.height())
            if current_pixmap_width_on_screen * factor < min(20, widget_min_dim * 0.1) : # Don't zoom out if image becomes too small
                 factor = min(20, widget_min_dim * 0.1) / current_pixmap_width_on_screen if current_pixmap_width_on_screen > 0 else 1.0


        new_scale_factor = self.scale_factor * factor
        new_scale_factor = clamp(new_scale_factor, min_scale_factor, max_scale_factor)


        self.scale_factor = new_scale_factor
        self._calculate_display_rect()
        new_disp_x, new_disp_y = self._image_to_display_coords(img_x, img_y)

        self.offset_x += anchor_point.x() - new_disp_x
        self.offset_y += anchor_point.y() - new_disp_y

        self._calculate_display_rect()
        self.update()


    def zoom_in(self, anchor_point: Optional[QPoint] = None):
        self.zoom(1.2, anchor_point)

    def zoom_out(self, anchor_point: Optional[QPoint] = None):
        self.zoom(1 / 1.2, anchor_point)

    def pan(self, delta_x: int, delta_y: int):
        """Pan the image by a delta."""
        if self.pixmap is None: return
        self.offset_x += delta_x
        self.offset_y += delta_y
        self._calculate_display_rect() 
        self.update()

    # --- Painting ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background_color)

        if self.pixmap is None or self.image_cv2_bgr is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        if self.display_rect.isNull() or self.display_rect.size() != self._get_target_display_size():
            self._calculate_display_rect()

        painter.drawPixmap(self.display_rect, self.pixmap, self.pixmap.rect())

        img_h, img_w = self.image_cv2_bgr.shape[:2]
        for i, annotation in enumerate(self.annotations):
            bbox_norm = annotation.get('bbox', [0, 0, 0, 0])
            class_id = annotation.get('class_id', -1) 

            x1, y1, x2, y2 = cxcywh_normalized_to_xyxy(bbox_norm, img_w, img_h)
            disp_x1, disp_y1 = self._image_to_display_coords(x1, y1)
            disp_x2, disp_y2 = self._image_to_display_coords(x2, y2)
            box_rect = QRect(disp_x1, disp_y1, disp_x2 - disp_x1, disp_y2 - disp_y1)

            is_selected = (i == self.selected_box_idx)
            is_unassigned = (class_id < 0 or class_id >= len(self.class_names))

            if is_unassigned:
                box_color = self.unassigned_color
                label_text = "Unassigned"
                pen_width = self.box_line_width
                pen_style = Qt.PenStyle.DotLine
            else:
                box_color = self._get_color_for_class(class_id)
                label_text = self.class_names[class_id]
                pen_width = self.box_line_width * 2 if is_selected else self.box_line_width
                pen_style = Qt.PenStyle.DashLine if is_selected else Qt.PenStyle.SolidLine

            confidence = annotation.get('confidence')
            if confidence is not None:
                 label_text += f" ({confidence:.2f})"

            pen = QPen(box_color, pen_width)
            pen.setStyle(pen_style)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush) 
            painter.drawRect(box_rect)

            if label_text:
                label_width = self.font_metrics.horizontalAdvance(label_text) + 10
                label_height = self.font_metrics.height() + 4
                label_x = box_rect.left()
                label_y = box_rect.top() - label_height if box_rect.top() >= label_height else box_rect.top()

                painter.fillRect(label_x, label_y, label_width, label_height, box_color)
                text_color = calculate_contrast_color(box_color)
                painter.setPen(text_color)
                painter.setFont(self.label_font)
                painter.drawText(
                    label_x + 5,
                    label_y + self.font_metrics.ascent() + 2, 
                    label_text
                )

        if self._drawing_box and self._start_drag_pos and self._current_drag_pos:
            draw_rect = QRect(self._start_drag_pos, self._current_drag_pos).normalized()
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(draw_rect)

        painter.end()

    # --- Mouse Events ---

    def mousePressEvent(self, event):
        pos = event.pos()
        self._last_mouse_pos = pos

        if self.pixmap is None: return

        if event.button() == Qt.MouseButton.RightButton:
             self.requestContextMenu.emit(pos) 
             return 

        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ControlModifier)):
            if self.display_rect.contains(pos):
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing_enabled:
            if self.display_rect.contains(pos):
                self._drawing_box = True
                self._start_drag_pos = pos
                self._current_drag_pos = pos
                self.select_box(-1) 
                self.update()
                return

        if event.button() == Qt.MouseButton.LeftButton and not self._is_drawing_enabled:
            clicked_box_idx = self._get_box_at(pos)
            if clicked_box_idx != self.selected_box_idx:
                self.select_box(clicked_box_idx) 
            elif clicked_box_idx == -1 and self.selected_box_idx != -1:
                 self.select_box(-1)
            return

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._last_mouse_pos is None: 
            self._last_mouse_pos = pos
            return

        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos

        if self._panning:
            self.pan(delta.x(), delta.y())
            return

        if self._drawing_box:
            self._current_drag_pos = pos
            self.update()
            return

        if self._is_drawing_enabled:
            if self.display_rect.contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.display_rect.contains(pos):
            hover_idx = self._get_box_at(pos)
            if hover_idx != -1:
                 self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                 self.setCursor(Qt.CursorShape.ArrowCursor) 
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)


    def mouseReleaseEvent(self, event):
        pos = event.pos()

        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and self._panning): 
            if self._panning:
                self._panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor) 
                self.mouseMoveEvent(event) 
                return

        if event.button() == Qt.MouseButton.LeftButton and self._drawing_box:
            if self._start_drag_pos is not None and self._current_drag_pos is not None:
                start_img_x, start_img_y = self._display_to_image_coords(self._start_drag_pos.x(), self._start_drag_pos.y())
                end_img_x, end_img_y = self._display_to_image_coords(self._current_drag_pos.x(), self._current_drag_pos.y())

                img_rect = QRect(QPoint(start_img_x, start_img_y), QPoint(end_img_x, end_img_y)).normalized()

                if img_rect.width() > 1 and img_rect.height() > 1:
                    self.newBoxDrawn.emit(img_rect)

            self._reset_drawing()
            self.update()
            return

    def wheelEvent(self, event):
        """Handle mouse wheel zooming."""
        if self.pixmap is None: return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier: 
             angle = event.angleDelta().y()
             if angle > 0:
                 self.zoom_in(event.position().toPoint())
             elif angle < 0:
                 self.zoom_out(event.position().toPoint())
             event.accept() 

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        self._update_timer.start()

    def keyPressEvent(self, event):
        """Handle key presses (e.g., for deleting boxes)."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.selected_box_idx != -1 and not self._is_drawing_enabled: # Only delete if not drawing
                self.deleteSelectionKeyPressed.emit()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore() 

    # --- Internal Helper Methods ---

    def _reset_interactions(self):
        """Reset flags related to mouse interactions."""
        self._panning = False
        self._drawing_box = False
        self._start_drag_pos = None
        self._current_drag_pos = None
        self._last_mouse_pos = None
        if not self._is_drawing_enabled:
             self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
             self.setCursor(Qt.CursorShape.CrossCursor)

    def _reset_drawing(self):
        """Reset only the drawing state variables."""
        self._drawing_box = False
        self._start_drag_pos = None
        self._current_drag_pos = None

    def _get_target_display_size(self) -> Tuple[int, int]:
        """Calculate the target size of the pixmap display area based on scale and aspect ratio."""
        if self.pixmap is None: return (0, 0)

        widget_w = self.width()
        widget_h = self.height()
        
        # Base size is the pixmap size scaled to fit the widget while maintaining aspect ratio
        pix_w_on_widget = widget_w
        pix_h_on_widget = pix_w_on_widget / self.aspect_ratio
        if pix_h_on_widget > widget_h:
            pix_h_on_widget = widget_h
            pix_w_on_widget = pix_h_on_widget * self.aspect_ratio
        
        # Apply overall scale_factor (zoom)
        scaled_w = pix_w_on_widget * self.scale_factor
        scaled_h = pix_h_on_widget * self.scale_factor
        
        return int(scaled_w), int(scaled_h)


    def _calculate_display_rect(self):
        """Calculate and cache the display rectangle, applying offsets and clamping."""
        if self.pixmap is None:
            self.display_rect = QRect()
            return

        widget_w = self.width()
        widget_h = self.height()
        scaled_w, scaled_h = self._get_target_display_size()

        if scaled_w <= 0 or scaled_h <= 0:
            self.display_rect = QRect()
            return

        base_x = (widget_w - scaled_w) / 2.0
        base_y = (widget_h - scaled_h) / 2.0
        
        # Clamp offsets
        # Max offset means image edge aligns with widget edge
        # Min offset means other image edge aligns with other widget edge
        
        # If image is wider than widget, allow panning until edges meet
        # max_abs_offset_x is how much the center can move from the true center
        max_abs_offset_x = (scaled_w - widget_w) / 2.0 if scaled_w > widget_w else 0
        max_abs_offset_y = (scaled_h - widget_h) / 2.0 if scaled_h > widget_h else 0

        self.offset_x = clamp(self.offset_x, -max_abs_offset_x, max_abs_offset_x)
        self.offset_y = clamp(self.offset_y, -max_abs_offset_y, max_abs_offset_y)
        
        final_x = base_x - self.offset_x # Offset moves the image relative to its centered position
        final_y = base_y - self.offset_y

        self.display_rect = QRect(int(final_x), int(final_y), int(scaled_w), int(scaled_h))


    def _image_to_display_coords(self, img_x: float, img_y: float) -> Tuple[int, int]:
        """Convert image pixel coordinates to display widget coordinates."""
        if self.image_cv2_bgr is None or self.display_rect.isNull():
            return (0, 0)

        img_h, img_w = self.image_cv2_bgr.shape[:2]
        if img_w == 0 or img_h == 0: return (0, 0)

        norm_x = img_x / img_w
        norm_y = img_y / img_h

        disp_x = self.display_rect.x() + norm_x * self.display_rect.width()
        disp_y = self.display_rect.y() + norm_y * self.display_rect.height()

        return int(disp_x), int(disp_y)

    def _display_to_image_coords(self, disp_x: int, disp_y: int) -> Tuple[int, int]:
        """Convert display widget coordinates to image pixel coordinates."""
        if self.image_cv2_bgr is None or self.display_rect.isNull() or \
           self.display_rect.width() == 0 or self.display_rect.height() == 0:
            return (0, 0)

        img_h, img_w = self.image_cv2_bgr.shape[:2]

        norm_x = (disp_x - self.display_rect.x()) / self.display_rect.width()
        norm_y = (disp_y - self.display_rect.y()) / self.display_rect.height()

        norm_x = clamp(norm_x, 0.0, 1.0)
        norm_y = clamp(norm_y, 0.0, 1.0)

        img_x = norm_x * img_w
        img_y = norm_y * img_h

        return int(img_x), int(img_y)

    def _get_box_at(self, pos: QPoint) -> int:
        """Find the index of the bounding box at the given display coordinates."""
        if self.image_cv2_bgr is None or not self.annotations:
            return -1

        img_h, img_w = self.image_cv2_bgr.shape[:2]

        for i in range(len(self.annotations) - 1, -1, -1):
            annotation = self.annotations[i]
            bbox_norm = annotation.get('bbox', [0, 0, 0, 0])

            x1, y1, x2, y2 = cxcywh_normalized_to_xyxy(bbox_norm, img_w, img_h)
            disp_x1, disp_y1 = self._image_to_display_coords(x1, y1)
            disp_x2, disp_y2 = self._image_to_display_coords(x2, y2)
            box_rect = QRect(disp_x1, disp_y1, disp_x2 - disp_x1, disp_y2 - disp_y1).normalized()

            if box_rect.contains(pos):
                return i
        return -1

    def _get_color_for_class(self, class_id: int) -> QColor:
        """Get a color for a class ID."""
        if 0 <= class_id < len(self.default_class_colors):
            return self.default_class_colors[class_id]
        else:
            h = (class_id * 37 + 50) % 360
            s = ((class_id * 61 + 150) % 106) + 150 
            v = ((class_id * 41 + 200) % 56) + 200   
            return QColor.fromHsv(h, s, v)