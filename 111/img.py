from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import torch

prompt = "a moonim dressed as a knight, riding a horse towards a medieval castle"

ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q3_K_S.gguf"
# ckpt_path = "/Volumes/SSD2TB/AI/caches/models/flux1-dev-Q2_K.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

height, width = 1024, 1024

images = pipeline(
    prompt=prompt,
    num_inference_steps=15,
    guidance_scale=5.0,
    height=height,
    width=width,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

images.save("gguf_image.png")