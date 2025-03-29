import sys
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                             QComboBox, QProgressBar, QMessageBox, QGroupBox,
                             QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ModelThread(QThread):
    """Thread for processing the T5 model without freezing the UI"""
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, input_text, task, max_length, num_beams, do_sample):
        super().__init__()
        self.input_text = input_text
        self.task = task
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample
        
    def run(self):
        try:
            self.progress_update.emit(10)
            
            # Load model and tokenizer
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
            self.progress_update.emit(30)
            
            model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
            self.progress_update.emit(50)
            
            # Format input with task prefix
            if self.task != "custom":
                # Add task prefix for standard tasks
                input_text = f"{self.task}: {self.input_text}"
            else:
                input_text = self.input_text
            
            # Tokenize input
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            self.progress_update.emit(70)
            
            # Generate output
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=self.do_sample,
                    early_stopping=True
                )
            self.progress_update.emit(90)
            
            # Decode output
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            self.progress_update.emit(100)
            
            self.result_ready.emit(output)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class T5ModelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("T5 Model Interface")
        self.setGeometry(100, 100, 800, 600)
        
        self.init_ui()
        
    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Title and description
        title_label = QLabel("T5 Model Text Processing")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        
        description = QLabel("This application uses the Google T5-base model to process text for various NLP tasks.")
        description.setWordWrap(True)
        
        # Task selection
        task_group = QGroupBox("Task Selection")
        task_layout = QVBoxLayout()
        
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "summarize", "translate English to German", 
            "translate English to French", "translate English to Romanian",
            "stsb", "cola", "custom"
        ])
        task_layout.addWidget(QLabel("Select Task:"))
        task_layout.addWidget(self.task_combo)
        task_group.setLayout(task_layout)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter your text here...")
        
        input_layout.addWidget(self.input_text)
        input_group.setLayout(input_layout)
        
        # Generation parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QHBoxLayout()
        
        # Max length
        max_length_layout = QVBoxLayout()
        max_length_layout.addWidget(QLabel("Max Length:"))
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(10, 512)
        self.max_length_spin.setValue(100)
        max_length_layout.addWidget(self.max_length_spin)
        
        # Num beams
        num_beams_layout = QVBoxLayout()
        num_beams_layout.addWidget(QLabel("Num Beams:"))
        self.num_beams_spin = QSpinBox()
        self.num_beams_spin.setRange(1, 10)
        self.num_beams_spin.setValue(4)
        num_beams_layout.addWidget(self.num_beams_spin)
        
        # Sampling checkbox
        sampling_layout = QVBoxLayout()
        sampling_layout.addWidget(QLabel("Sampling:"))
        self.do_sample_check = QCheckBox("Enable sampling")
        sampling_layout.addWidget(self.do_sample_check)
        
        params_layout.addLayout(max_length_layout)
        params_layout.addLayout(num_beams_layout)
        params_layout.addLayout(sampling_layout)
        params_group.setLayout(params_layout)
        
        # Process button
        self.process_button = QPushButton("Process Text")
        self.process_button.setMinimumHeight(40)
        self.process_button.clicked.connect(self.process_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Output section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        
        # Add all widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(description)
        main_layout.addWidget(task_group)
        main_layout.addWidget(input_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(self.process_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(output_group)
        
        # Set main layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def process_text(self):
        input_text = self.input_text.toPlainText()
        
        if not input_text.strip():
            QMessageBox.warning(self, "Warning", "Please enter some text to process.")
            return
        
        self.statusBar().showMessage("Processing...")
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        task = self.task_combo.currentText()
        max_length = self.max_length_spin.value()
        num_beams = self.num_beams_spin.value()
        do_sample = self.do_sample_check.isChecked()
        
        # Create and start worker thread
        self.worker = ModelThread(input_text, task, max_length, num_beams, do_sample)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.start()
    
    def on_result_ready(self, result):
        self.output_text.setText(result)
        self.statusBar().showMessage("Processing complete")
        self.process_button.setEnabled(True)
    
    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.statusBar().showMessage("Error occurred")
        self.process_button.setEnabled(True)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = T5ModelGUI()
    window.show()
    sys.exit(app.exec_())