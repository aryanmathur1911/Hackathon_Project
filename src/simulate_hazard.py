import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np

class RoadHazardSimulator:
    def __init__(self):
        # Optimized for T4 GPU
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")

    def create_hazard_mask(self, width, height):
        """Creates a 'target zone' mask in the lower third of the road."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        # Coordinates focused on the driving lane
        draw.rectangle([width*0.2, height*0.65, width*0.8, height*0.85], fill=255)
        return mask

    def simulate(self, image_path, hazard_type="pothole", output_path="simulated_road.jpg"):
        init_image = Image.open(image_path).convert("RGB").resize((512, 512))
        mask_image = self.create_hazard_mask(512, 512)
        
        prompts = {
            "pothole": "a deep jagged asphalt pothole, highly realistic, road damage",
            "tree": "a large fallen tree trunk blocking the asphalt road",
            "animal": "a stray dog sitting in the middle of the road"
        }

        print(f"Generating {hazard_type} simulation...")
        generated_image = self.pipe(
            prompt=prompts.get(hazard_type, prompts["pothole"]),
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=30
        ).images[0]
        
        generated_image.save(output_path)
        return output_path