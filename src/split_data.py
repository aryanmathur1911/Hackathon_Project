import splitfolders
from pathlib import Path

def split_dataset():
    # Define your paths
    input_folder = Path("data/synthetic/images")
    output_folder = Path("data/processed")

    # Check if input exists
    if not input_folder.exists():
        print(f"❌ Error: {input_folder} does not exist. Run generate.py first!")
        return

    print(f"Spliting data from {input_folder}...")

    # splitfolders.ratio does the following:
    # 1. Creates 'train' and 'val' folders in 'data/processed'
    # 2. Keeps the 'Broken' and 'Not Broken' subfolder structure
    # 3. Randomly shuffles the images
    # Ratio (0.8, 0.2) means 80% Train, 20% Val
    splitfolders.ratio(
        input_folder, 
        output=str(output_folder), 
        seed=42, 
        ratio=(.8, .2), 
        group_prefix=None, 
        move=False # Set to True if you want to move instead of copy
    )

    print(f"✅ Success! Data split into:")
    print(f" - {output_folder}/train")
    print(f" - {output_folder}/val")

if __name__ == "__main__":
    split_dataset()