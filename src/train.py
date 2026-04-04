import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Configuration
DATA_DIR = Path("data/processed")
MODEL_SAVE_PATH = "models/road_classifier.pth"
Path("models").mkdir(exist_ok=True)

# 2. Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15 # Increased slightly to account for the harder shadow training
LEARNING_RATE = 0.0001 # Lower learning rate for better stability with augmentations

# 3. Shadow-Proof Preprocessing Logic
class ShadowProofDataset(Dataset):
    """Custom Dataset to apply CLAHE and Albumentations to our images."""
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def apply_clahe(self, image):
        # Convert to LAB to equalize lighting (Shadow Eraser)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Shadow Eraser first
        image = self.apply_clahe(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# 4. Advanced Data Augmentation
# This is the secret sauce: we force the AI to see 'Fake Shadows' on safe roads
train_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def train_model():
    # Load Datasets using our Custom ShadowProof Class
    train_set = ShadowProofDataset(DATA_DIR / "train", transform=train_transform)
    val_set = ShadowProofDataset(DATA_DIR / "val", transform=val_transform)
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Shadow-Proof Model on: {device}")

    # Load Pretrained ResNet-18
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # [Hazard, Safe]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save the FULL model state (weights)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Shadow-Proof Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()