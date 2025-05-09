import os
import sys # For redirecting output, potentially
import traceback
from ultralytics import YOLO # type: ignore
# import torch # For checking CUDA availability, etc. (optional here, YOLO might handle it)

# It's good practice to ensure the environment for ultralytics is set up correctly,
# especially if running in a frozen app or complex environment.
# For now, we assume standard Python execution environment.

class TrainingLogger:
    """
    A simple class to capture stdout/stderr during training.
    This can be passed to a QThread to emit signals with the log messages.
    For now, it just prints.
    """
    def __init__(self, log_signal_emitter=None):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_signal_emitter = log_signal_emitter # e.g., a pyqtSignal(str)

    def write(self, message):
        # self.original_stdout.write(message) # Still print to console for debugging
        if self.log_signal_emitter:
            self.log_signal_emitter.emit(message.strip())
        else:
            # Fallback if no signal emitter is provided
            self.original_stdout.write(message) 

    def flush(self):
        # self.original_stdout.flush()
        pass # sys.stdout methods might not be fully compatible with all objects

    def __enter__(self):
        # sys.stdout = self
        # sys.stderr = self # Capture stderr as well
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # sys.stdout = self.original_stdout
        # sys.stderr = self.original_stderr
        pass # For now, don't redirect, ultralytics callback system is better

def start_yolo_training(params: dict, progress_callback=None, log_callback=None) -> dict:
    """
    Starts the YOLO model training process.

    Args:
        params (dict): A dictionary containing training parameters, e.g.:
            {
                "model": "yolov8n.pt",  // or path to custom.pt
                "data": "/path/to/dataset.yaml",
                "epochs": 100,
                "batch": 16,
                "imgsz": 640,
                "device": "cpu", // or "0" for GPU 0
                "project": "runs/train", // Main output directory
                "name": "exp" // Sub-directory for this run
                # ... other ultralytics supported params
            }
        progress_callback (callable, optional): A function to call with epoch progress.
                                                Receives (current_epoch, total_epochs, metrics_dict).
        log_callback (callable, optional): A function to call with log messages from training.
                                           Receives (log_message_string).

    Returns:
        dict: A dictionary containing results, e.g.:
            {
                "success": True/False,
                "message": "Training completed successfully.",
                "best_model_path": "/path/to/runs/train/exp/weights/best.pt",
                "error": "Error message if any"
            }
    """
    results = {"success": False, "message": "", "best_model_path": None, "error": None}

    model_name = params.get("model", "yolov8n.pt")
    data_yaml = params.get("data")
    epochs = params.get("epochs", 100)
    batch = params.get("batch", 16)
    imgsz = params.get("imgsz", 640)
    device = params.get("device", "cpu")
    project_dir = params.get("project", "runs/train")
    run_name = params.get("name", "exp")
    
    # Ultralytics callbacks
    _callbacks = {}
    if progress_callback:
        def on_epoch_end(trainer): # trainer is ultralytics.engine.trainer.BaseTrainer
            # trainer.epoch is current epoch (0-indexed when this is called after epoch finishes)
            # trainer.epochs is total epochs (1-indexed)
            # metrics are available in trainer.metrics or trainer.validator.metrics
            # For simplicity, we'll pass basic epoch info.
            # Note: epoch in trainer is 0-indexed, so add 1 for 1-indexed display
            current_e = trainer.epoch + 1
            total_e = trainer.epochs
            # Try to get some common metrics, may vary based on task (detect, segment, classify)
            metrics_to_log = {}
            if hasattr(trainer, 'metrics') and trainer.metrics:
                 metrics_to_log = {k: v for k,v in trainer.metrics.items() if isinstance(v, (float, int))} # log main metrics
            
            # If you need specific metrics like mAP50-95, you might need to dig into trainer.validator.metrics
            # For example, for detection:
            # if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
            #    map50_95 = trainer.validator.metrics.get('metrics/mAP50-95(B)', 0.0) # (B) for box AP
            #    metrics_to_log['mAP50-95'] = map50_95
            
            progress_callback(current_e, total_e, metrics_to_log)
        _callbacks["on_epoch_end"] = on_epoch_end

    if log_callback:
        # Ultralytics has its own logger, trying to capture all stdout/stderr can be tricky
        # and might interfere with its internal logging.
        # A more robust way is to use its callback system if it provides direct log access,
        # or parse specific events. For now, this is a placeholder.
        # If ultralytics.utils.LOGGER is easily configurable, that's one way.
        # For now, we'll rely on the on_fit_epoch_end or other granular callbacks for info.
        # The TrainingLogger class above is an alternative if direct output capture is needed.
        
        # Example for specific log points if available via callbacks:
        def on_train_start(trainer):
            log_callback(f"Training started. Model: {model_name}, Data: {data_yaml}, Epochs: {epochs}, Device: {device}")
            log_callback(f"Output will be saved to: {os.path.join(project_dir, run_name)}")
        _callbacks["on_train_start"] = on_train_start
        
        def on_train_end(trainer):
            log_callback("Training finished.")
        _callbacks["on_train_end"] = on_train_end
        
        def on_fit_epoch_end(trainer): # Called after on_epoch_end, has more complete metrics usually
             if log_callback and hasattr(trainer, 'metrics_pretty'):
                  log_callback(f"Epoch {trainer.epoch + 1}/{trainer.epochs} Summary: {trainer.metrics_pretty}")
        _callbacks["on_fit_epoch_end"] = on_fit_epoch_end


    try:
        print(f"[YOLO Trainer] Initializing model: {model_name}")
        model = YOLO(model_name)
        
        # Add registered callbacks
        for event, func in _callbacks.items():
            model.add_callback(event, func)

        print(f"[YOLO Trainer] Starting training with parameters: {params}")
        # The train method itself will print a lot to stdout
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project_dir,
            name=run_name,
            exist_ok=True, # Allow overwriting if run_name exists, or set to False
            # Other parameters can be added here as needed, e.g.,
            # workers=8, patience=50, optimizer='AdamW', lr0=0.001, etc.
        )
        
        # Ultralytics automatically saves the best model in project/name/weights/best.pt
        # and last model in project/name/weights/last.pt
        best_model_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            results["success"] = True
            results["message"] = "Training completed successfully."
            results["best_model_path"] = os.path.abspath(best_model_path)
            print(f"[YOLO Trainer] Training successful. Best model: {best_model_path}")
        else:
            results["message"] = "Training completed, but 'best.pt' not found."
            print(f"[YOLO Trainer] Training completed, but 'best.pt' not found in expected location: {best_model_path}")
            # Check for last.pt as a fallback
            last_model_path = os.path.join(project_dir, run_name, 'weights', 'last.pt')
            if os.path.exists(last_model_path):
                 results["success"] = True # Still consider it a success if last.pt is there
                 results["best_model_path"] = os.path.abspath(last_model_path) # Report last.pt
                 results["message"] += f" Using 'last.pt' as fallback: {last_model_path}"


    except Exception as e:
        error_info = traceback.format_exc()
        results["success"] = False
        results["message"] = f"An error occurred during training: {e}"
        results["error"] = error_info
        print(f"[YOLO Trainer] Error during training: {e}\n{error_info}")
        if log_callback:
            log_callback(f"TRAINING ERROR: {e}\n{error_info}")


    finally:
        # Remove callbacks if added, to prevent issues if model object is reused (though typically not)
        for event in _callbacks.keys():
            if hasattr(model, 'clear_callbacks'): # Newer ultralytics versions might have this
                 model.clear_callbacks(event) 
            elif hasattr(model, 'callbacks') and event in model.callbacks:
                 model.callbacks[event] = [] # Older way to clear (be cautious)
        if log_callback: # Final log for the calling thread
            log_callback(f"[YOLO Trainer] Final result: Success: {results['success']}, Msg: {results['message']}")


    return results

if __name__ == '__main__':
    # --- Example Usage (for testing this module directly) ---
    
    # Create a dummy dataset.yaml for testing
    dummy_data_yaml_content = f"""
train: {os.path.abspath('./dummy_dataset/images/train')}
val: {os.path.abspath('./dummy_dataset/images/val')}
# test: {os.path.abspath('./dummy_dataset/images/test')} # Optional

nc: 2
names: ['class1', 'class2']
"""
    dummy_yaml_path = os.path.abspath("./dummy_dataset/dataset.yaml")
    os.makedirs(os.path.dirname(dummy_yaml_path), exist_ok=True)
    with open(dummy_yaml_path, 'w') as f:
        f.write(dummy_data_yaml_content)
    
    # Create dummy image directories (content not essential for trainer to start, but paths must exist)
    os.makedirs("./dummy_dataset/images/train", exist_ok=True)
    os.makedirs("./dummy_dataset/images/val", exist_ok=True)
    # You would normally populate these with images and label files.

    print(f"Dummy dataset.yaml created at: {dummy_yaml_path}")
    print(f"Ensure you have 'yolov8n.pt' or specify another model.")
    print("If testing with GPU, ensure PyTorch with CUDA is installed and CUDA drivers are correct.")

    test_params = {
        "model": "yolov8n.pt",  # Downloaded automatically by ultralytics if not present
        "data": dummy_yaml_path,
        "epochs": 3,  # Keep low for testing
        "batch": 2,   # Keep low for testing
        "imgsz": 64, # Keep very low for speed testing
        "device": "cpu", # Use "0" for GPU if available and configured
        "project": "runs/test_train",
        "name": "exp_trainer_test",
        "patience": 3 # Early stopping for testing
    }

    def _test_progress_cb(epoch, total_epochs, metrics):
        print(f"  [Test CB] Progress: Epoch {epoch}/{total_epochs}, Metrics: {metrics}")

    def _test_log_cb(message):
        print(f"  [Test Log CB] {message}")

    print(f"\nStarting test training with params:\n{test_params}")
    
    training_results = start_yolo_training(test_params, progress_callback=_test_progress_cb, log_callback=_test_log_cb)
    
    print(f"\n--- Training Test Finished ---")
    print(f"Success: {training_results.get('success')}")
    print(f"Message: {training_results.get('message')}")
    print(f"Best Model Path: {training_results.get('best_model_path')}")
    if training_results.get('error'):
        print(f"Error Details:\n{training_results.get('error')}")
    
    # Cleanup dummy files (optional)
    # import shutil
    # if os.path.exists("./dummy_dataset"):
    #     shutil.rmtree("./dummy_dataset")
    # if os.path.exists(test_params["project"]): # Be careful with this if you have other runs
    #      pass # shutil.rmtree(test_params["project"]) 