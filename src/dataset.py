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
        
        self.image_dir = os.path.join(data_root, mode, "image")
        self.pose_dir = os.path.join(data_root, mode, "pose") # New Folder
        
        self.image_files = sorted(os.listdir(self.image_dir))
        
        # Transform: Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Mask Transform: No Normalization, just 0-1
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def get_upper_body_mask(self, img_size):
        w, h = img_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([w//4, h//4, w*3//4, h*3//4], fill=255)
        return mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # 2. Load PRE-COMPUTED Pose
        # Note: We saved poses as .png, images might be .jpg
        pose_name = img_name.replace(".jpg", ".png")
        pose_path = os.path.join(self.pose_dir, pose_name)
        
        if os.path.exists(pose_path):
            pose = Image.open(pose_path).convert("RGB")
        else:
            # Fallback (Should not happen if you ran preprocess)
            pose = Image.new("RGB", (512, 512), (0, 0, 0))

        # 3. Create Mask
        mask = self.get_upper_body_mask(image.size)
        
        # 4. Returns
        return {
            "pixel_values": self.transform(image),
            "pose_values": self.transform(pose), # Already resized/normalized
            "mask": self.mask_transform(mask),
            "filename": img_name
        }