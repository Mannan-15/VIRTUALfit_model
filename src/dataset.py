import os
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VITONDataset(Dataset):
    def __init__(self, data_root, mode="train"):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        
        # Paths to subfolders
        self.image_dir = os.path.join(data_root, mode, "image")
        self.cloth_dir = os.path.join(data_root, mode, "cloth")
        
        # Read the pair list (e.g., "00001_00.jpg 00001_00.jpg")
        # If pairs file doesn't exist, we just match by filename
        self.image_files = sorted(os.listdir(self.image_dir))
        
        # Transforms (Resize to 512x512 and Normalize to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        # Mask transform (No normalization, just 0-1)
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def get_upper_body_mask(self, img_size):
        """
        Creates a simple box mask for the upper body.
        In a full Phase 2 app, we would use a Segmentation Model here.
        """
        w, h = img_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Approximate torso box (Left, Top, Right, Bottom)
        draw.rectangle([w//4, h//4, w*3//4, h*3//4], fill=255)
        return mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 1. Load Person Image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # 2. Load Cloth Image (Assumes same filename)
        cloth_path = os.path.join(self.cloth_dir, img_name)
        if not os.path.exists(cloth_path):
            # Fallback: Try to find any cloth if exact match fails (rare)
            cloth_path = img_path 
        cloth = Image.open(cloth_path).convert("RGB")

        # 3. Create Mask (The area we want to repaint)
        mask = self.get_upper_body_mask(image.size)
        
        # 4. Apply Transforms
        pixel_values = self.transform(image)
        cloth_values = self.transform(cloth)
        mask_values = self.mask_transform(mask)

        return {
            "pixel_values": pixel_values, # The Person
            "cloth_values": cloth_values, # The Cloth (Target)
            "mask": mask_values,          # The Hole
            "filename": img_name
        }
