"""Lab 19: Image captioning workflow with BLIP and optional Stable Diffusion.

Important distinction: Stable Diffusion is a text-to-image diffusion model, not an
image-to-text captioner. This practical uses BLIP to caption an input image, then
optionally sends that caption to Stable Diffusion to generate a new image.
Install optional packages: pip install transformers diffusers accelerate
"""
import os
import torch
from PIL import Image, ImageDraw
from transformers import BlipProcessor, BlipForConditionalGeneration

vin_device = "cuda" if torch.cuda.is_available() else "cpu"
vin_image_path = "image.jpg"
vin_generate_image_with_sd = False  # Change to True to run the optional diffusion step.

# ---------- Load an input image; create a simple placeholder if one is not provided
if os.path.exists(vin_image_path):
    vin_image = Image.open(vin_image_path).convert("RGB")
else:
    vin_image = Image.new("RGB", (256, 256), (80, 140, 190))
    vin_draw = ImageDraw.Draw(vin_image)
    vin_draw.ellipse((60, 60, 190, 190), fill=(220, 90, 60))
    print("image.jpg not found; using a synthetic image for a pipeline demonstration.")

# ---------- Image -> text caption using BLIP
vin_model_name = "Salesforce/blip-image-captioning-base"
vin_processor = BlipProcessor.from_pretrained(vin_model_name)
vin_model = BlipForConditionalGeneration.from_pretrained(vin_model_name).to(vin_device)
vin_inputs = vin_processor(images=vin_image, return_tensors="pt").to(vin_device)
with torch.no_grad():
    vin_output = vin_model.generate(**vin_inputs, max_new_tokens=30)
vin_caption = vin_processor.decode(vin_output[0], skip_special_tokens=True)
print("Generated caption:", vin_caption)

# ---------- Optional: text caption -> new image using Stable Diffusion
# This does not create the caption; it creates an image from the text caption.
if vin_generate_image_with_sd:
    from diffusers import StableDiffusionPipeline
    vin_sd_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vin_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    vin_sd = StableDiffusionPipeline.from_pretrained(vin_sd_name, torch_dtype=vin_dtype)
    vin_sd = vin_sd.to(vin_device)
    vin_generated = vin_sd(prompt=vin_caption, num_inference_steps=20).images[0]
    vin_generated.save("19_stable_diffusion_generated.png")
    print("Saved generated image as 19_stable_diffusion_generated.png")
