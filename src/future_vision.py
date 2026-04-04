import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

class TextureVisionAnalyzer:
    def __init__(self, model_path="models/road_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the architecture
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Load the weights (mapping to current device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Hook into the last convolutional layer for the Heatmap (Grad-CAM)
        self.model.layer4.register_forward_hook(self.save_gradient)
        self.gradients = None

    def save_gradient(self, module, input, output):
        self.gradients = output

    def apply_shadow_eraser(self, img_cv):
        """Uses CLAHE to even out lighting and neutralize soft shadows."""
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # clipLimit=3.0 helps brighten shadows without washing out the whole road
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def get_texture_map(self, img_cv):
        """Converts image to a Sobel edge map to focus on physical roughness."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Sobel detects sharp changes in intensity (pothole edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255 range
        if np.max(mag) > 0:
            texture = np.uint8(255 * mag / np.max(mag))
        else:
            texture = np.zeros_like(gray, dtype=np.uint8)
            
        return cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)

    def visualize(self, image_path, output_path="spatial_fix_analysis.jpg"):
        img_cv = cv2.imread(image_path)
        h, w, _ = img_cv.shape

        # 1. PRE-PROCESS (Same as before)
        clean_img = self.apply_shadow_eraser(img_cv)
        texture_img = self.get_texture_map(clean_img)
        
        # 2. THE SPATIAL MASK (NEW)
        # Create a triangle mask that only looks at the road in front
        mask = np.zeros((h, w), dtype=np.uint8)
        # Defining a trapezoid that covers the lanes and ignores the sky/trees
        # Adjust these points if your camera angle is different
        roi_corners = np.array([[(0, h), (w//2, int(h*0.45)), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
        
        # Black out everything outside the road before the AI predicts
        masked_texture = cv2.bitwise_and(texture_img, texture_img, mask=mask)

        # 3. PREDICT ON MASKED IMAGE
        pil_img = Image.fromarray(masked_texture)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)

        # 4. HEATMAP GENERATION
        output = self.model(input_tensor)
        score = output[:, 0] 
        self.model.zero_grad()
        score.backward()

        grads = self.gradients.cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(grads.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * grads[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (w, h))
        
        # Multiply heatmap by the mask so the 'Sky Glow' is erased
        cam = cam * (mask / 255.0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 5. FINAL BLEND
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result = cv2.addWeighted(img_cv, 0.7, heatmap, 0.3, 0)
        
        cv2.imwrite(output_path, result)
        print(f"✅ Spatial-filtered analysis saved to {output_path}")

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    analyzer = TextureVisionAnalyzer()
    # Replace 'input.jpg' with your filename
    analyzer.visualize("clean_road.jpeg", "final_result.jpg")