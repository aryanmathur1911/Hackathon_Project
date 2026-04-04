import cv2
import numpy as np

def shadow_eraser_preprocessor(image_path):
    # 1. Load image and convert to HSV
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This "flattens" the lighting, making dark shadows look like bright road
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_equalized = clahe.apply(v)

    # 3. Merge back and convert to RGB
    hsv_equalized = cv2.merge((h, s, v_equalized))
    result_rgb = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)
    
    return result_rgb