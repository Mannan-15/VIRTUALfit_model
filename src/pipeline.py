import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from controlnet_aux import DWposeDetector

class LiteMPVTONPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading DWPose Detector...")
        self.dwpose = DWposeDetector.from_pretrained(
            "yzd-v/DWPose", cache_dir="./models/dwpose"
        ).to(device)
        
    def get_pose_tensor(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        pose = self.dwpose(img, detect_resolution=512, image_resolution=512)
        # Convert to tensor (B, C, H, W) range [-1, 1]
        pose_tensor = torch.tensor(np.array(pose)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return pose_tensor.to(self.device)

    def prepare_inputs(self, vae, image_path, mask_path):
        # Load and process image/mask
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # VAE Encoding
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215
            
        return latents, mask_tensor
