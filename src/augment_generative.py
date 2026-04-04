import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import os
from pathlib import Path

class ObstacleGenerator:
    def __init__(self):
        # 1. Setup paths
        self.source_dir = Path("data/processed/train/Not Broken")
        self.output_dir = Path("data/processed/train/Broken")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load Model to Colab T4 GPU
        print("Loading Stable Diffusion (this takes a minute)...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")

    def create_road_mask(self, width, height):
        """Creates a mask in the lower-middle part of the image (the road)."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        # Target the lower half of the image where obstacles usually sit
        # [left, top, right, bottom]
        draw.rectangle([width*0.2, height*0.6, width*0.8, height*0.9], fill=255)
        return mask

    def run_generation(self, prompt, num_images=10):
        images = list(self.source_dir.glob("*.jpg"))[:num_images]
        
        print(f"Generating '{prompt}' on {len(images)} images...")
        
        for i, img_path in enumerate(images):
            init_image = Image.open(img_path).convert("RGB").resize((512, 512))
            mask_image = self.create_road_mask(512, 512)

            # Generate the new image
            output = self.pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=25
            ).images[0]

            # Save to the 'Broken' folder
            save_name = f"gen_{prompt.replace(' ', '_')}_{i}.jpg"
            output.save(self.output_dir / save_name)
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    gen = ObstacleGenerator()
    
    # Run different "Flywheel" scenarios
    gen.run_generation(prompt="a large deep pothole with cracks on asphalt road", num_images=5)
    gen.run_generation(prompt="a fallen tree trunk blocking a paved road", num_images=5)
    gen.run_generation(prompt="a large stray dog sitting in the middle of the road", num_images=5)