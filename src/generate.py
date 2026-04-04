import cv2
import albumentations as A
import os
from pathlib import Path

class SyntheticGenerator:
    def __init__(self):
        # Input: data/raw/Road Classification/Broken (or Not Broken)
        self.raw_path = Path("data/raw/Road Classification")
        
        # Output: data/synthetic/images/Broken (or Not Broken)
        self.out_images_path = Path("data/synthetic/images")
        
        # Create the base output folder
        self.out_images_path.mkdir(parents=True, exist_ok=True)

    def apply_augmentations(self, image):
        """Augmentations tailored for road surface classification."""
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3), 
            A.RandomGamma(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ])
        return transform(image=image)['image']

    def save_synthetic(self, image, class_name, original_name):
        """Saves image into data/synthetic/images/[ClassName]"""
        # Create class subfolder inside 'images'
        class_folder = self.out_images_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        
        # Final path: data/synthetic/images/Broken/synth_photo.jpg
        save_path = class_folder / f"synth_{original_name}"
        
        # OpenCV needs a string path
        success = cv2.imwrite(str(save_path), image)
        return success

if __name__ == "__main__":
    gen = SyntheticGenerator()
    
    # Exact folder names from your Kaggle dataset
    classes = ["Broken", "Not Broken"]
    total_processed = 0

    print(f"🔍 Checking Input Path: {gen.raw_path.absolute()}")
    print(f"📁 Saving Output To: {gen.out_images_path.absolute()}")

    for category in classes:
        category_path = gen.raw_path / category
        
        if not category_path.exists():
            print(f"❌ Error: Folder NOT found: {category_path}")
            continue

        # Case-insensitive search for images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            images.extend(list(category_path.glob(ext)))
            
        print(f"✅ Found {len(images)} images in '{category}'")

        for img_path in images:
            image = cv2.imread(str(img_path))
            
            if image is not None:
                # Apply the augmentation
                augmented_img = gen.apply_augmentations(image)
                
                # Save to synthetic/images/[Category]
                success = gen.save_synthetic(augmented_img, category, img_path.name)
                
                if success:
                    total_processed += 1
            else:
                print(f"⚠️ Could not read: {img_path.name}")

    print(f"\n--- Process Complete ---")
    print(f"Total synthetic images generated: {total_processed}")