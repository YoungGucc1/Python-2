import gradio as gr
import torch
# Try to import the specific model class if available, otherwise use the generic one
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    ModelClass = Qwen2VLForConditionalGeneration
    print("Using specific Qwen2VLForConditionalGeneration class")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoProcessor
    ModelClass = AutoModelForCausalLM
    print("Using generic AutoModelForCausalLM class")

from qwen import process_vision_info
from PIL import Image
import numpy as np
import os
from datetime import datetime

def array_to_image_path(image_array):
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    # Convert numpy array to PIL Image
    img = Image.fromarray(np.uint8(image_array))
    
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    
    # Save the image
    img.save(filename)
    
    # Get the full path of the saved image
    full_path = os.path.abspath(filename)
    
    return full_path
    
models = {}
processors = {}

DESCRIPTION = "[Qwen2.5-VL-7B Demo](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)"

# Initialize model and processor when they're first needed to save memory
def get_model_and_processor(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    global models, processors
    
    if model_id not in models:
        print(f"Loading model {model_id}...")
        models[model_id] = ModelClass.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        ).cuda().eval()
        
        processors[model_id] = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
    return models[model_id], processors[model_id]

user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"

def run_example(image, text_input=None, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    if image is None:
        return "Please upload an image first."
    
    if text_input is None or text_input.strip() == "":
        text_input = "What's in this image?"
    
    try:
        image_path = array_to_image_path(image)
        model, processor = get_model_and_processor(model_id)
        
        image = Image.fromarray(image).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clean up the temporary image file
        try:
            os.remove(image_path)
        except:
            pass
            
        return output_text[0]
    except Exception as e:
        return f"Error: {str(e)}"

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Qwen2.5-VL-7B Input"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                model_selector = gr.Dropdown(choices=["Qwen/Qwen2.5-VL-7B-Instruct"], label="Model", value="Qwen/Qwen2.5-VL-7B-Instruct")
                text_input = gr.Textbox(label="Question", placeholder="What's in this image?")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text", lines=10)

        submit_btn.click(run_example, [input_img, text_input, model_selector], [output_text])

if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(debug=True)