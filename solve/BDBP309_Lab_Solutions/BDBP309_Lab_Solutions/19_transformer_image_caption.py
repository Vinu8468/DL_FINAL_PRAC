"""Additional material listed after Lab 19: Transformer image captioning.

Uses a pretrained Vision Transformer encoder + GPT-2 decoder captioning model.
Install: pip install transformers pillow
The pretrained model is downloaded on first run.
"""
import os
import torch
from PIL import Image, ImageDraw
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

vin_device = "cuda" if torch.cuda.is_available() else "cpu"
vin_model_name = "nlpconnect/vit-gpt2-image-captioning"
vin_processor = ViTImageProcessor.from_pretrained(vin_model_name)
vin_tokenizer = AutoTokenizer.from_pretrained(vin_model_name)
vin_model = VisionEncoderDecoderModel.from_pretrained(vin_model_name).to(vin_device)

if os.path.exists("image.jpg"):
    vin_image = Image.open("image.jpg").convert("RGB")
else:
    vin_image = Image.new("RGB", (224, 224), (90, 140, 170))
    vin_draw = ImageDraw.Draw(vin_image)
    vin_draw.rectangle((50, 50, 170, 170), fill=(220, 100, 60))
    print("image.jpg not found; using a synthetic placeholder image.")

vin_pixel_values = vin_processor(images=vin_image, return_tensors="pt").pixel_values.to(vin_device)
with torch.no_grad():
    vin_output_ids = vin_model.generate(vin_pixel_values, max_new_tokens=30, num_beams=4)
vin_caption = vin_tokenizer.decode(vin_output_ids[0], skip_special_tokens=True)
print("Transformer caption:", vin_caption)
