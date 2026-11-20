# pip install torch transformers pillow


import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

#------------ Load model

vin_device = "cuda" if torch.cuda.is_available() else "cpu"

vin_model_name = "Salesforce/blip-image-captioning-base"

vin_processor = BlipProcessor.from_pretrained(vin_model_name)

vin_model = BlipForConditionalGeneration.from_pretrained(
    vin_model_name
).to(vin_device)

#------------ Load image

vin_image = Image.open("image.jpg").convert("RGB")

#------------ Preprocess image

vin_inputs = vin_processor(
    images=vin_image,
    return_tensors="pt"
).to(vin_device)

#------------ Generate caption

with torch.no_grad():
    vin_output = vin_model.generate(
        **vin_inputs,
        max_new_tokens=30
    )

#------------ Convert output to text

vin_caption = vin_processor.decode(
    vin_output[0],
    skip_special_tokens=True
)

print("Generated Caption:", vin_caption)


# pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline

#------------ Load Stable Diffusion

vin_sd = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

vin_sd = vin_sd.to(vin_device)

#------------ Generate image using caption

vin_generated_image = vin_sd(
    prompt=vin_caption,
    num_inference_steps=20
).images[0]

vin_generated_image.save("generated_image.png")

vin_generated_image.show()
