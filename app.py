import os
import base64
from io import BytesIO
import gradio as gr
from mistralai import Mistral
from PIL import Image
from pathlib import Path

api_key = os.environ.get("MISTRAL")
client = Mistral(api_key=api_key)

#config
VALID_DOCUMENT_EXTENSIONS = {".pdf"}
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png",}

def upload_pdf(content, filename):
    uploaded_file = client.files.upload(
        file={"file_name": filename, "content": content},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    return signed_url.url
def process_ocr(document_source):
    return client.ocr.process(
        model="mistral-ocr-latest",
        document=document_source,
        include_image_base64=True
    )
def do_ocr(input_type, url=None, file=None):
    document_source = None
    
    if input_type == "URL":
        if not url or url.strip() == "":
            return "Please provide a valid URL.", "", []
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS):
            document_source = {"type": "image_url", "image_url": url.strip()}
        else:
            document_source = {"type": "document_url", "document_url": url.strip()}

    elif input_type == "Upload file":
        if not file:
            return "Please upload a file.", "", []
        file_name = file.name.lower()
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in VALID_DOCUMENT_EXTENSIONS:
            with open(file.name, "rb") as f:
                content = f.read()
            signed_url = upload_pdf(content, os.path.basename(file_name))
            document_source = {"type": "document_url", "document_url": signed_url}
        elif file_extension in VALID_IMAGE_EXTENSIONS:
            img = Image.open(file)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            document_source = {"type": "image_url", "image_url": f"data:image/png;base64,{img_str}"}
        else:
            return f"Error: Unsupported file type. Supported types: {', '.join(VALID_DOCUMENT_EXTENSIONS | VALID_IMAGE_EXTENSIONS)}", "", []

    else:
        return "Invalid input type ", "", []

    ocr_response = process_ocr(document_source)
    markdown_text = "\n\n".join(page.markdown for page in ocr_response.pages)
    extracted_text = markdown_text
    rendered_markdown = markdown_text
    images = []

    for page in ocr_response.pages:
        for img in page.images:
            if img.image_base64:
                base64_str = img.image_base64
                if "," in base64_str:
                    base64_str = base64_str.split(",")[1]
                img_bytes = base64.b64decode(base64_str)
                img_pil = Image.open(BytesIO(img_bytes))
                images.append(img_pil)
                img_buffer = BytesIO()
                img_pil.save(img_buffer, format="PNG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                data_url = f"data:image/png;base64,{img_base64}"
                rendered_markdown = rendered_markdown.replace(
                    f"![{img.id}]({img.id})", f"![{img.id}]({data_url})"
                )
            else:
                rendered_markdown += f"\n\n[Image Warning: No base64 data for {img.id}]"

    return extracted_text.strip(), rendered_markdown.strip(), images
        
custom_css = """
    body {font-family: body {font-family: 'Helvetica Neue', Helvetica;}
    .gr-button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px;}
    .gr-button:hover {background-color: #45a049;}
    .gr-textbox {margin-bottom: 15px;}
    .example-button {background-color: #1E90FF; color: white; border: none; padding: 8px 15px; border-radius: 5px; margin: 5px;}
    .example-button:hover {background-color: #FF4500;}
    .tall-radio .gr-radio-item {padding: 15px 0; min-height: 50px; display: flex; align-items: center;}
    .tall-radio label {font-size: 16px;}
"""
with gr.Blocks(
    title="Mistral OCR Demo",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #333;'>Mistral OCR Demo</h1>")
    gr.Markdown("<p style='text-align: center; color: #666;'>Extract text and images from PDFs or images using Mistral's latest OCR model. You can also see markdown live.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                choices=["URL", "Upload file"],
                label="Input Type",
                value="URL",
                elem_classes="tall-radio"
            )
            url_input = gr.Textbox(
                label="Document or Image URL",
                placeholder="e.g., https://arxiv.org/pdf/2501.12948",
                visible=True,
                lines=1
            )
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                visible=False
            )
            submit_btn = gr.Button("Extract Text and Images")
            gr.Markdown("### Try These Examples")
            pdf_example = gr.Button("PDF", elem_classes="example-button")
            img_example = gr.Button("Image", elem_classes="example-button")
        with gr.Column(scale=2):
            cleaned_output = gr.Textbox(label="Extracted Plain Text", lines=10, show_copy_button=True)
            markdown_output = gr.Markdown(label="Rendered Markdown Text")
            image_output = gr.Gallery(label="OCR Extracted Images", columns=2, height="auto")

    def update_visibility(choice):
        return gr.update(visible=(choice == "URL")), gr.update(visible=(choice == "Upload file"))
    input_type.change(fn=update_visibility, inputs=input_type, outputs=[url_input, file_input])
    def set_url_and_type(url):
        return url, "URL"

    pdf_example.click(
        fn=lambda: set_url_and_type("https://arxiv.org/pdf/2501.12948"),
        outputs=[url_input, input_type]
    )
    img_example.click(
        fn=lambda: set_url_and_type("https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit0/recommended-pace.jpg"),
        outputs=[url_input, input_type]
    )

    submit_btn.click(
        fn=do_ocr,
        inputs=[input_type, url_input, file_input],
        outputs=[cleaned_output, markdown_output, image_output]
    )
    
if __name__ == "__main__":
    demo.launch()
