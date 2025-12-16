import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
# Switch to the stable OpenposeDetector
from controlnet_aux import OpenposeDetector

class LiteMPVTONPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading OpenPose Detector...")
        # OpenPose is standard and always works with from_pretrained
        self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.pose_detector.to(device)
        
    def get_pose_tensor(self, image_path):
        """Extracts Pose Skeleton and converts to Tensor (B, C, H, W)"""
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        
        # Detect Pose (Enable hands and face for better detail)
        # result is a PIL Image of the skeleton
        pose = self.pose_detector(img, include_body=True, include_hand=True, include_face=True)
        
        # Normalize to [-1, 1] for VAE
        # Pose image is usually black background (0) with colored limbs
        pose_tensor = torch.tensor(np.array(pose)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return pose_tensor.to(self.device)

    def prepare_mask(self, image_path):
        """Creates a simple rectangular mask for testing."""
        # In Phase 2, this will be replaced by an AI Segmentation Model
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        # Draw white rectangle (masked area) over torso
        draw.rectangle([(128, 100), (384, 400)], fill=255)
        return mask

    def prepare_inputs(self, vae, image_path):
        """Encodes Image and Mask for the Model"""
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        mask = self.prepare_mask(image_path)
        
        # Image -> Tensor [-1, 1]
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        # Mask -> Tensor [0, 1]
        mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # VAE Encoding (Latents)
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215
            
        return latents, mask_tensor