import sys
import os
import io
import random
import copy
import datetime
import traceback

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QComboBox, QLineEdit, QProgressBar, QMessageBox,
    QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openpyxl # For Excel output
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# --- Model Configuration ---
# Use a dictionary to store model/processor info for easier management
MODEL_INFO = {
    'microsoft/Florence-2-large-ft': {'model': None, 'processor': None},
    'microsoft/Florence-2-large': {'model': None, 'processor': None},
    'microsoft/Florence-2-base-ft': {'model': None, 'processor': None},
    'microsoft/Florence-2-base': {'model': None, 'processor': None},
}

# Quantization config (optional, helps reduce memory footprint)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Core Processing Logic (Adapted from Gradio app) ---

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def load_model_and_processor(model_id):
    """Loads a specific model and processor if not already loaded."""
    if MODEL_INFO[model_id]['model'] is None:
        print(f"Loading model: {model_id}...")
        try:
            # Add quantization_config=bnb_config here if using BitsAndBytes
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                # quantization_config=bnb_config # Uncomment if using quantization
            ).to(DEVICE).eval()
            MODEL_INFO[model_id]['model'] = model
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise
    if MODEL_INFO[model_id]['processor'] is None:
        print(f"Loading processor: {model_id}...")
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            MODEL_INFO[model_id]['processor'] = processor
        except Exception as e:
            print(f"Error loading processor {model_id}: {e}")
            raise
    return MODEL_INFO[model_id]['model'], MODEL_INFO[model_id]['processor']

def run_inference(model, processor, task_prompt, image, text_input=None):
    """Runs the Florence-2 model inference."""
    if text_input is None or text_input.strip() == "":
        prompt = task_prompt
    else:
        # Ensure there's a space if text_input is provided for tasks expecting it
        prompt = task_prompt + " " + text_input if task_prompt.endswith('>') else task_prompt + text_input

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        # Use torch.inference_mode() for efficiency
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(DEVICE), # Ensure pixels are on correct device
                max_new_tokens=1024,
                early_stopping=False, # Keep False unless specific need
                do_sample=False,      # Keep False for deterministic output
                num_beams=3,
            )
        # Ensure generated_ids are moved to CPU before decoding if they aren't already
        generated_text = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=False)[0]

        # Debug: Print raw generated text
        # print(f"Raw generated text: {generated_text}")

        # Determine the actual task used for post-processing
        # Some tasks implicitly change (e.g., Caption + Grounding becomes <CAPTION_TO_PHRASE_GROUNDING> later)
        # We need the *final* task string the model expects for post-processing
        final_task_str = task_prompt.split(" ")[0] # Get the <TASK_TAG> part
        # print(f"Using task for post-processing: {final_task_str}")


        parsed_answer = processor.post_process_generation(
            generated_text,
            task=final_task_str,
            image_size=(image.width, image.height)
        )

        # Debug: Print parsed answer
        # print(f"Parsed answer: {parsed_answer}")

        return parsed_answer, final_task_str # Return the final task used as well

    except Exception as e:
        print(f"Error during inference or post-processing: {e}")
        print(f"Task Prompt: {task_prompt}")
        print(f"Generated Text: {generated_text if 'generated_text' in locals() else 'N/A'}")
        # traceback.print_exc() # Print detailed traceback
        # Return a minimal error structure
        return {"error": str(e)}, task_prompt.split(" ")[0]


# --- Image Annotation Functions (Adapted) ---

def plot_bbox(image, data):
    """Draws bounding boxes on the image using Matplotlib."""
    # Convert PIL image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    # Ensure 'bboxes' and 'labels' keys exist and are lists
    bboxes = data.get('bboxes', [])
    labels = data.get('labels', [])
    if not isinstance(bboxes, list): bboxes = []
    if not isinstance(labels, list): labels = []

    # Make sure labels list matches bbox list length
    if len(labels) < len(bboxes):
        labels.extend(["N/A"] * (len(bboxes) - len(labels)))
    elif len(labels) > len(bboxes):
        labels = labels[:len(bboxes)]


    for bbox, label in zip(bboxes, labels):
         # Check if bbox is valid (should be 4 coordinates)
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Clamp coordinates to image bounds to prevent errors
            h, w = image_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            # Ensure width and height are positive
            if x2 > x1 and y2 > y1:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(x1, y1 - 5, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.7))
        else:
             print(f"Skipping invalid bbox format: {bbox}")
    ax.axis('off')
    # Convert Matplotlib figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def draw_polygons(image, prediction, fill_mask=False):
    """Draws polygons on the image using PIL."""
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    scale = 1
    polygons_data = prediction.get('polygons', [])
    labels = prediction.get('labels', [])
    if not isinstance(polygons_data, list): polygons_data = []
    if not isinstance(labels, list): labels = []

    # Adjust labels length
    if len(labels) < len(polygons_data):
        labels.extend(["N/A"] * (len(polygons_data) - len(labels)))
    elif len(labels) > len(polygons_data):
        labels = labels[:len(polygons_data)]

    for polygons, label in zip(polygons_data, labels):
        color = random.choice(colormap)
        # Ensure polygons is a list of lists (list of polygons for one label)
        if not isinstance(polygons, list):
            print(f"Skipping invalid polygon data structure: {polygons}")
            continue

        for _polygon in polygons:
            # Ensure _polygon is a list or tuple of coordinates
            if not isinstance(_polygon, (list, tuple)):
                 print(f"Skipping invalid polygon structure: {_polygon}")
                 continue
            _polygon_np = np.array(_polygon).reshape(-1, 2)
            if len(_polygon_np) < 3:
                # print('Skipping invalid polygon (less than 3 points):', _polygon_np)
                continue

            _polygon_flat = (_polygon_np * scale).reshape(-1).tolist()

            try:
                if fill_mask:
                     # Use a semi-transparent fill
                    fill_color_rgba = Image.new("RGBA", output_image.size, (0, 0, 0, 0))
                    draw_fill = ImageDraw.Draw(fill_color_rgba)
                    fill_rgb = ImageColor.getrgb(color)
                    draw_fill.polygon(_polygon_flat, outline=color, fill=fill_rgb + (100,)) # Add alpha
                    output_image = Image.alpha_composite(output_image.convert("RGBA"), fill_color_rgba).convert("RGB")
                    # Redraw outline after compositing
                    draw = ImageDraw.Draw(output_image)
                    draw.polygon(_polygon_flat, outline=color, width=2)
                else:
                    draw.polygon(_polygon_flat, outline=color, width=2)

                # Draw text label - find a good spot (e.g., top-left point)
                if _polygon_flat:
                    text_x, text_y = _polygon_flat[0], _polygon_flat[1]
                    # Basic check to keep text within bounds
                    text_x = max(0, min(text_x, output_image.width - 20))
                    text_y = max(0, min(text_y, output_image.height - 10))
                    draw.text((text_x + 5, text_y + 2), label, fill='white', font=ImageFont.load_default()) # Simple white text

            except Exception as draw_err:
                print(f"Error drawing polygon: {draw_err}, polygon data: {_polygon_flat}")

    return output_image


def convert_to_od_format(data):
    """Extracts bboxes and labels for OD-like plotting."""
    bboxes = data.get('bboxes', [])
    # Handle potential variations in label key names
    labels = data.get('labels', data.get('bboxes_labels', []))
    return {'bboxes': bboxes, 'labels': labels}

def draw_ocr_bboxes(image, prediction):
    """Draws OCR bounding boxes (quadrilaterals) on the image."""
    output_image = image.copy()
    scale = 1
    draw = ImageDraw.Draw(output_image)
    # Ensure keys exist and are lists
    bboxes = prediction.get('quad_boxes', [])
    labels = prediction.get('labels', [])
    if not isinstance(bboxes, list): bboxes = []
    if not isinstance(labels, list): labels = []

    # Adjust label length
    if len(labels) < len(bboxes):
        labels.extend(["N/A"] * (len(bboxes) - len(labels)))
    elif len(labels) > len(bboxes):
        labels = labels[:len(bboxes)]


    for box, label in zip(bboxes, labels):
         # Check if box is valid (list of 8 coordinates or 4 pairs)
        if isinstance(box, list) and len(box) == 8:
             color = random.choice(colormap)
             new_box = (np.array(box) * scale).tolist()
             try:
                 draw.polygon(new_box, width=2, outline=color)
                 # Position text near the first point, slightly offset
                 text_x, text_y = new_box[0], new_box[1]
                 text_x = max(0, min(text_x, output_image.width - 20))
                 text_y = max(0, min(text_y - 10, output_image.height - 10)) # Move above the box
                 draw.text((text_x, text_y),
                           f"{label}",
                           fill=color,
                           font=ImageFont.load_default())
             except Exception as draw_err:
                 print(f"Error drawing OCR box: {draw_err}, box data: {new_box}")
        else:
            print(f"Skipping invalid OCR box format: {box}")
    return output_image

# --- Worker Thread for Batch Processing ---

class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)             # Signal for progress bar (0-100)
    status_update = pyqtSignal(str)           # Signal for status messages
    result_ready = pyqtSignal(dict)           # Signal for individual results (optional display)
    processing_finished = pyqtSignal(list)    # Signal when all processing is done (sends all results)
    error_signal = pyqtSignal(str)            # Signal for errors during processing

    def __init__(self, image_paths, model_id, task_key, text_input, output_dir):
        super().__init__()
        self.image_paths = image_paths
        self.model_id = model_id
        self.task_key = task_key # The key selected in the dropdown (e.g., 'Caption', 'Object Detection')
        self.text_input = text_input
        self.output_dir = output_dir
        self.is_cancelled = False
        self._model = None
        self._processor = None

    def run(self):
        results_list = []
        total_images = len(self.image_paths)

        try:
            # Load model and processor once for the batch
            self.status_update.emit(f"Loading model/processor: {self.model_id}...")
            self._model, self._processor = load_model_and_processor(self.model_id)
            self.status_update.emit(f"Model/processor loaded. Starting batch...")

            for i, img_path in enumerate(self.image_paths):
                if self.is_cancelled:
                    self.status_update.emit("Processing cancelled.")
                    break

                filename = os.path.basename(img_path)
                self.status_update.emit(f"Processing {i+1}/{total_images}: {filename}")

                try:
                    image = Image.open(img_path).convert("RGB") # Ensure RGB
                except Exception as e:
                    self.error_signal.emit(f"Error opening image {filename}: {e}")
                    results_list.append({
                        'image_path': img_path,
                        'task': self.task_key,
                        'result_text': f"Error loading image: {e}",
                        'annotated_image_path': None,
                        'error': str(e)
                    })
                    self.progress_update.emit(int((i + 1) * 100 / total_images))
                    continue # Skip to next image

                # --- Determine Task Prompt and Handle Cascaded Tasks ---
                task_prompt = ""
                current_text_input = self.text_input # Use provided text input initially
                requires_text_input = False
                processed_image = None # To store annotated image
                result_text_display = "N/A" # Text to show/save

                # Map dropdown key to actual task tag(s)
                if self.task_key == 'Caption':
                    task_prompt = '<CAPTION>'
                elif self.task_key == 'Detailed Caption':
                    task_prompt = '<DETAILED_CAPTION>'
                elif self.task_key == 'More Detailed Caption':
                    task_prompt = '<MORE_DETAILED_CAPTION>'
                elif self.task_key == 'Object Detection':
                    task_prompt = '<OD>'
                elif self.task_key == 'Dense Region Caption':
                    task_prompt = '<DENSE_REGION_CAPTION>'
                elif self.task_key == 'Region Proposal':
                    task_prompt = '<REGION_PROPOSAL>'
                elif self.task_key == 'Caption to Phrase Grounding':
                    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
                    requires_text_input = True
                elif self.task_key == 'Referring Expression Segmentation':
                    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
                    requires_text_input = True
                elif self.task_key == 'Region to Segmentation':
                    task_prompt = '<REGION_TO_SEGMENTATION>'
                    requires_text_input = True # Region specified as text "[x1, y1, x2, y2]"
                elif self.task_key == 'Open Vocabulary Detection':
                    task_prompt = '<OPEN_VOCABULARY_DETECTION>'
                    requires_text_input = True # Text is the vocabulary
                elif self.task_key == 'Region to Category':
                    task_prompt = '<REGION_TO_CATEGORY>'
                    requires_text_input = True # Region specified as text "[x1, y1, x2, y2]"
                elif self.task_key == 'Region to Description':
                    task_prompt = '<REGION_TO_DESCRIPTION>'
                    requires_text_input = True # Region specified as text "[x1, y1, x2, y2]"
                elif self.task_key == 'OCR':
                    task_prompt = '<OCR>'
                elif self.task_key == 'OCR with Region':
                    task_prompt = '<OCR_WITH_REGION>'
                # Cascaded tasks require multiple steps
                elif self.task_key == 'Caption + Grounding':
                    task_prompt = '<CAPTION>'
                elif self.task_key == 'Detailed Caption + Grounding':
                    task_prompt = '<DETAILED_CAPTION>'
                elif self.task_key == 'More Detailed Caption + Grounding':
                    task_prompt = '<MORE_DETAILED_CAPTION>'
                else:
                    self.error_signal.emit(f"Unknown task: {self.task_key}")
                    continue

                if requires_text_input and not current_text_input:
                    self.error_signal.emit(f"Task '{self.task_key}' requires text input, but none provided for {filename}.")
                    results_list.append({
                        'image_path': img_path,
                        'task': self.task_key,
                        'result_text': "Error: Task requires text input.",
                        'annotated_image_path': None,
                        'error': "Missing text input"
                    })
                    self.progress_update.emit(int((i + 1) * 100 / total_images))
                    continue

                # --- Run Inference (Potentially Multiple Steps) ---
                try:
                    # Handle cascaded tasks first
                    intermediate_result = None
                    if self.task_key in ['Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding']:
                        # Step 1: Get Caption
                        caption_task = task_prompt # e.g., <CAPTION>
                        caption_result_data, _ = run_inference(self._model, self._processor, caption_task, image)
                        if "error" in caption_result_data:
                             raise ValueError(f"Caption step failed: {caption_result_data['error']}")

                        intermediate_result = caption_result_data.get(caption_task, "")
                        if not intermediate_result:
                            raise ValueError("Caption step did not return expected text.")

                        # Step 2: Grounding
                        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
                        current_text_input = intermediate_result # Use the generated caption as input
                        result_data, final_task_used = run_inference(self._model, self._processor, task_prompt, image, current_text_input)
                        # Add caption back for context
                        if isinstance(result_data, dict):
                            result_data[caption_task] = intermediate_result
                    else:
                        # Single step task
                        result_data, final_task_used = run_inference(self._model, self._processor, task_prompt, image, current_text_input)

                    if "error" in result_data:
                        raise ValueError(f"Inference failed: {result_data['error']}")

                    # --- Process Results and Generate Output ---
                    task_output = result_data.get(final_task_used) # Use the actual task key returned

                    if task_output is None:
                         # Fallback or error message if the expected key isn't there
                         result_text_display = f"Error: No output found for task key '{final_task_used}'"
                         print(f"Warning: Output key '{final_task_used}' not found in result_data for {filename}. Full result: {result_data}")
                    elif isinstance(task_output, str):
                        # Simple text output (e.g., Caption)
                        result_text_display = task_output
                    elif isinstance(task_output, dict):
                        # Complex output (bboxes, polygons, etc.)
                        result_text_display = str(task_output) # Save the dict representation to Excel for now

                        # --- Generate Annotated Image (if applicable) ---
                        annotated_image = None
                        img_copy_for_annotation = image.copy()

                        try:
                            if final_task_used in ['<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<CAPTION_TO_PHRASE_GROUNDING>']:
                                annotated_image = plot_bbox(img_copy_for_annotation, task_output)
                            elif final_task_used == '<OPEN_VOCABULARY_DETECTION>':
                                bbox_results = convert_to_od_format(task_output)
                                annotated_image = plot_bbox(img_copy_for_annotation, bbox_results)
                            elif final_task_used in ['<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>']:
                                annotated_image = draw_polygons(img_copy_for_annotation, task_output, fill_mask=True)
                            elif final_task_used == '<OCR_WITH_REGION>':
                                annotated_image = draw_ocr_bboxes(img_copy_for_annotation, task_output)
                        except Exception as viz_error:
                             print(f"Error creating annotated image for {filename}: {viz_error}")
                             self.error_signal.emit(f"Couldn't generate visualization for {filename}: {viz_error}")

                        # Save annotated image if created
                        if annotated_image:
                            base, ext = os.path.splitext(filename)
                            annotated_filename = f"{base}_annotated_{final_task_used.strip('<>')}{ext}"
                            annotated_image_path = os.path.join(self.output_dir, annotated_filename)
                            try:
                                annotated_image.save(annotated_image_path)
                                processed_image = annotated_image_path
                            except Exception as save_error:
                                print(f"Error saving annotated image {annotated_filename}: {save_error}")
                                self.error_signal.emit(f"Couldn't save annotated image for {filename}: {save_error}")
                    else:
                        result_text_display = f"Unexpected result type: {type(task_output)}"


                except Exception as e:
                    self.error_signal.emit(f"Error processing {filename}: {e}")
                    # traceback.print_exc() # Optional: print full traceback to console
                    results_list.append({
                        'image_path': img_path,
                        'task': self.task_key,
                        'result_text': f"Error: {e}",
                        'annotated_image_path': None,
                        'error': str(e)
                    })
                    self.progress_update.emit(int((i + 1) * 100 / total_images))
                    continue # Skip to next image

                # Append successful result
                results_list.append({
                    'image_path': img_path,
                    'task': self.task_key,
                    'result_text': result_text_display,
                    'annotated_image_path': processed_image, # Path or None
                    'error': None
                })

                # Emit progress
                self.progress_update.emit(int((i + 1) * 100 / total_images))

                # Optional: Emit individual result for potential live display (not implemented in UI)
                # self.result_ready.emit(results_list[-1])

        except Exception as thread_e:
            self.error_signal.emit(f"Critical error in processing thread: {thread_e}")
            # traceback.print_exc() # Optional: print full traceback to console
        finally:
            if not self.is_cancelled:
                self.processing_finished.emit(results_list)
            # Clean up GPU memory if model was loaded in this thread (optional)
            # if self._model:
            #     del self._model
            # if self._processor:
            #     del self._processor
            # if DEVICE == "cuda":
            #     torch.cuda.empty_cache()

    def stop(self):
        self.is_cancelled = True
        self.status_update.emit("Cancellation requested...")

# --- Main Application Window ---

class FlorenceBatchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Florence-2 Batch Processor")
        self.setGeometry(100, 100, 900, 700) # x, y, width, height

        self.image_paths = []
        self.output_dir = ""
        self.processing_thread = None

        self.init_ui()
        self.apply_stylesheet()
        # Pre-load default model? Or load on demand? Loading on demand might be better for memory.
        # self.preload_default_model()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Controls ---
        top_controls_layout = QHBoxLayout()

        # Model Selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_INFO.keys())
        self.model_combo.setCurrentIndex(0) # Default to large-ft
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_controls_layout.addLayout(model_layout)

        # Task Selection
        task_layout = QVBoxLayout()
        task_label = QLabel("Select Task:")
        self.task_combo = QComboBox()
        # Combined list for simplicity, added prefixes for clarity
        all_tasks = [
            'Caption', 'Detailed Caption', 'More Detailed Caption',
            'Object Detection', 'Dense Region Caption', 'Region Proposal',
            'OCR', 'OCR with Region',
            '[Text Req] Caption to Phrase Grounding',
            '[Text Req] Referring Expression Segmentation',
            '[Text Req] Region to Segmentation',
            '[Text Req] Open Vocabulary Detection',
            '[Text Req] Region to Category',
            '[Text Req] Region to Description',
            '[Cascaded] Caption + Grounding',
            '[Cascaded] Detailed Caption + Grounding',
            '[Cascaded] More Detailed Caption + Grounding'
        ]
        self.task_combo.addItems(all_tasks)
        self.task_combo.setCurrentIndex(0) # Default to Caption
        self.task_combo.currentTextChanged.connect(self.update_text_input_visibility)
        task_layout.addWidget(task_label)
        task_layout.addWidget(self.task_combo)
        top_controls_layout.addLayout(task_layout)

        # Optional Text Input
        text_input_layout = QVBoxLayout()
        self.text_input_label = QLabel("Text Input (Required for some tasks):")
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter text here if required by the task...")
        text_input_layout.addWidget(self.text_input_label)
        text_input_layout.addWidget(self.text_input)
        top_controls_layout.addLayout(text_input_layout)

        main_layout.addLayout(top_controls_layout)

        # Initial visibility check for text input
        self.update_text_input_visibility(self.task_combo.currentText())


        # --- File Selection ---
        file_selection_layout = QHBoxLayout()
        self.select_images_btn = QPushButton("Select Images")
        self.select_images_btn.clicked.connect(self.select_images)
        self.image_list_widget = QListWidget()
        self.image_list_widget.setFixedHeight(150) # Limit height
        file_selection_layout.addWidget(self.select_images_btn)
        file_selection_layout.addWidget(self.image_list_widget)
        main_layout.addLayout(file_selection_layout)

        # --- Output Directory ---
        output_dir_layout = QHBoxLayout()
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output_dir)
        self.output_dir_label = QLineEdit()
        self.output_dir_label.setReadOnly(True)
        self.output_dir_label.setPlaceholderText("Select directory to save Excel results and annotated images...")
        output_dir_layout.addWidget(self.select_output_btn)
        output_dir_layout.addWidget(self.output_dir_label)
        main_layout.addLayout(output_dir_layout)

        # --- Processing Controls ---
        process_controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setObjectName("StartButton") # For styling
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        process_controls_layout.addWidget(self.start_btn)
        process_controls_layout.addWidget(self.cancel_btn)
        process_controls_layout.addWidget(self.progress_bar)
        main_layout.addLayout(process_controls_layout)

        # --- Status Area ---
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #cccccc; font-style: italic;")
        self.status_log = QTextEdit() # For more detailed log
        self.status_log.setReadOnly(True)
        self.status_log.setFixedHeight(100)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.status_log)

        self.setLayout(main_layout)

    def apply_stylesheet(self):
        # Simple modern stylesheet
        stylesheet = """
            QWidget {
                background-color: #2c3e50; /* Dark blue-gray */
                color: #ecf0f1; /* Light gray */
                font-size: 10pt;
            }
            QLabel {
                color: #bdc3c7; /* Lighter gray for labels */
                padding: 2px;
            }
            QPushButton {
                background-color: #3498db; /* Bright blue */
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9; /* Darker blue */
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
            QPushButton:disabled {
                background-color: #7f8c8d; /* Gray when disabled */
                color: #bdc3c7;
            }
             /* Specific style for Start button */
            QPushButton#StartButton {
                background-color: #2ecc71; /* Green */
            }
            QPushButton#StartButton:hover {
                background-color: #27ae60; /* Darker green */
            }
            QPushButton#StartButton:pressed {
                background-color: #1e8449;
            }

            QComboBox {
                background-color: #34495e; /* Slightly lighter dark blue */
                border: 1px solid #7f8c8d;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                 border: none;
            }
             QComboBox::down-arrow {
                 image: url(down_arrow.png); /* Optional: add a custom arrow icon */
             }

            QLineEdit {
                background-color: #34495e;
                border: 1px solid #7f8c8d;
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #34495e;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                alternate-background-color: #3a5064; /* Slightly different for alternating rows */
            }
            QProgressBar {
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                text-align: center;
                background-color: #34495e;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db; /* Blue progress */
                border-radius: 3px;
                 margin: 0.5px; /* Small margin around the chunk */
            }
            QTextEdit {
                 background-color: #1f2b38; /* Even darker for log */
                 border: 1px solid #7f8c8d;
                 color: #bdc3c7;
                 border-radius: 3px;
            }
        """
        self.setStyleSheet(stylesheet)

    # def preload_default_model(self):
    #     default_model_id = self.model_combo.currentText()
    #     try:
    #         self.update_status(f"Pre-loading default model: {default_model_id}...")
    #         QApplication.processEvents() # Keep UI responsive during load
    #         load_model_and_processor(default_model_id)
    #         self.update_status("Default model loaded.")
    #     except Exception as e:
    #         self.show_error_message(f"Failed to pre-load model {default_model_id}: {e}")
    #         self.update_status("Error loading default model.", is_error=True)

    def update_text_input_visibility(self, task_text):
        """Show/hide text input based on selected task."""
        requires_text = "[Text Req]" in task_text or "[Cascaded]" in task_text
        self.text_input_label.setVisible(requires_text)
        self.text_input.setVisible(requires_text)
        if not requires_text:
            self.text_input.clear() # Clear if not needed

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "", # Start directory
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if files:
            self.image_paths = files
            self.image_list_widget.clear()
            self.image_list_widget.addItems([os.path.basename(f) for f in files])
            self.update_status(f"Selected {len(files)} images.")

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(directory)
            self.update_status(f"Output directory set to: {directory}")

    def update_status(self, message, is_error=False):
        """Updates the status label and log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.status_log.append(log_message)
        self.status_label.setText(f"Status: {message}")
        if is_error:
            self.status_label.setStyleSheet("color: #e74c3c;") # Red for errors
        else:
            self.status_label.setStyleSheet("color: #cccccc; font-style: italic;") # Default style


    def show_error_message(self, message):
        """Displays an error message box."""
        QMessageBox.critical(self, "Error", message)
        self.update_status(f"Error: {message}", is_error=True)


    def start_processing(self):
        if not self.image_paths:
            self.show_error_message("Please select images first.")
            return
        if not self.output_dir:
            self.show_error_message("Please select an output directory first.")
            return

        selected_model = self.model_combo.currentText()
        selected_task_key = self.task_combo.currentText().split('] ')[-1] # Get actual task name
        text_input = self.text_input.text().strip()

        # Check if text input is required but not provided
        if "[Text Req]" in self.task_combo.currentText() and not text_input:
            self.show_error_message(f"Task '{selected_task_key}' requires text input.")
            return

        # Disable start, enable cancel, reset progress
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.select_images_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.task_combo.setEnabled(False)
        self.text_input.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_log.clear() # Clear log for new batch

        self.update_status("Starting processing...")

        # Create and start the processing thread
        self.processing_thread = ProcessingThread(
            self.image_paths,
            selected_model,
            selected_task_key,
            text_input,
            self.output_dir
        )

        # Connect signals
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.status_update.connect(lambda msg: self.update_status(msg))
        self.processing_thread.error_signal.connect(lambda err: self.update_status(err, is_error=True))
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.finished.connect(self.on_thread_finished) # Qt signal when thread run() exits

        self.processing_thread.start()


    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.update_status("Attempting to cancel...")
            self.cancel_btn.setEnabled(False) # Prevent multiple clicks

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_processing_finished(self, results_list):
        """Called when the thread finishes processing all items."""
        self.update_status(f"Processing complete. Processed {len(results_list)} items.")
        # Save results to Excel
        if results_list:
            try:
                excel_path = self.save_to_excel(results_list)
                self.update_status(f"Results saved to: {excel_path}")
                QMessageBox.information(self, "Processing Complete", f"Batch processing finished.\nResults saved to:\n{excel_path}")
            except Exception as e:
                self.show_error_message(f"Failed to save results to Excel: {e}")
        else:
             QMessageBox.warning(self, "Processing Complete", "Processing finished, but no results were generated.")

        # Don't re-enable controls here, wait for on_thread_finished

    def on_thread_finished(self):
        """Called when the QThread object itself has finished execution."""
        self.update_status("Thread finished.")
        self.processing_thread = None # Clean up thread reference

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_images_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.task_combo.setEnabled(True)
        self.text_input.setEnabled(True)
        self.progress_bar.setValue(0) # Reset progress bar


    def save_to_excel(self, results):
        """Saves the results list to an Excel file."""
        if not self.output_dir or not results:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"florence2_results_{timestamp}.xlsx"
        excel_filepath = os.path.join(self.output_dir, excel_filename)

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Florence-2 Results"

        # Headers
        headers = ["Image Filename", "Full Image Path", "Task", "Text Result", "Annotated Image Path", "Error"]
        sheet.append(headers)

        # Apply basic styling to header
        for cell in sheet[1]:
            cell.font = openpyxl.styles.Font(bold=True)
            cell.fill = openpyxl.styles.PatternFill("solid", fgColor="DDDDDD") # Light gray fill

        # Data
        for result in results:
            row_data = [
                os.path.basename(result.get('image_path', 'N/A')),
                result.get('image_path', 'N/A'),
                result.get('task', 'N/A'),
                str(result.get('result_text', '')), # Ensure it's a string
                result.get('annotated_image_path', 'N/A'),
                result.get('error', '')
            ]
            sheet.append(row_data)

        # Adjust column widths (optional, can be slow for many columns/rows)
        for col in sheet.columns:
             max_length = 0
             column = col[0].column_letter # Get column name like 'A'
             for cell in col:
                 try: # Necessary to avoid error on empty cells
                     if len(str(cell.value)) > max_length:
                         max_length = len(str(cell.value))
                 except:
                     pass
             adjusted_width = (max_length + 2)
             # Set a max width to prevent extremely wide columns
             sheet.column_dimensions[column].width = min(adjusted_width, 60)


        workbook.save(excel_filepath)
        return excel_filepath

    def closeEvent(self, event):
        """Handle closing the window, especially if processing is running."""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "Processing is still running. Are you sure you want to exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.cancel_processing() # Attempt graceful cancellation
                # Optionally wait a short time for the thread to potentially finish
                self.processing_thread.wait(1000) # Wait up to 1 second
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlorenceBatchApp()
    window.show()
    sys.exit(app.exec())