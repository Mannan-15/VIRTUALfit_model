import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from controlnet_aux import OpenposeDetector

# --- CONFIG ---
DATA_ROOT = "data/viton_hd"
SPLIT = "train"

def preprocess():
    # 1. Setup paths
    img_dir = os.path.join(DATA_ROOT, SPLIT, "image")
    save_dir = os.path.join(DATA_ROOT, SPLIT, "pose")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Load Detector (On GPU for speed)
    print("Loading OpenPose...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)

    # 3. Process Images
    images = sorted(os.listdir(img_dir))
    print(f"Found {len(images)} images. Generating poses...")
    
    for img_name in tqdm(images):
        save_path = os.path.join(save_dir, img_name.replace(".jpg", ".png"))
        
        # Skip if already done
        if os.path.exists(save_path):
            continue
            
        try:
            # Load and Resize
            img_path = os.path.join(img_dir, img_name)
            image = Image.open(img_path).convert("RGB").resize((512, 512))
            
            # Detect Pose
            pose = detector(image, include_body=True, include_hand=True, include_face=True)
            
            # Save as PNG
            pose.save(save_path)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Done! Poses saved.")

if __name__ == "__main__":
    preprocess()
