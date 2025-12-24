import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class VITONDataset(Dataset):
    def __init__(self, data_root, mode="train"):
        self.data_root = data_root
        self.mode = mode
        
        # Determine the pair file (train_pairs.txt or test_pairs.txt)
        pair_file = os.path.join(data_root, f"{mode}_pairs.txt")
        with open(pair_file, "r") as f:
            self.pairs = f.read().splitlines()
            
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Parse the line: "person_A.jpg person_B.jpg"
        line = self.pairs[index]
        person_name, source_name = line.split()

        # 1. Define Paths
        # TARGET (Person A)
        image_path = os.path.join(self.data_root, self.mode, "image", person_name)
        mask_path = os.path.join(self.data_root, self.mode, "agnostic-mask", person_name.replace(".jpg", "_mask.png"))
        
        # Pose (For Person A)
        pose_name = person_name.replace(".jpg", "_rendered.png")
        pose_path = os.path.join(self.data_root, self.mode, "openpose_img", pose_name)
        
        # SOURCE (Person B) - CHANGED: We now look in the 'image' folder too!
        source_path = os.path.join(self.data_root, self.mode, "image", source_name)

        # 2. Load & Process Person A (Target)
        image = Image.open(image_path).convert("RGB").resize((384, 512))
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image.transpose(2, 0, 1)) # [3, H, W]

        # 3. Load & Process Mask (Target)
        mask = Image.open(mask_path).convert("L").resize((384, 512), resample=Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0) # [1, H, W]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # 4. Load & Process Pose (Target)
        pose = Image.open(pose_path).convert("RGB").resize((384, 512))
        pose = np.array(pose).astype(np.float32) / 127.5 - 1.0
        pose = torch.from_numpy(pose.transpose(2, 0, 1)) # [3, H, W]

        # 5. Load & Process SOURCE (Person B)
        # We treat the *entire photo* of Person B as the "cloth" input.
        # The model will learn to extract the texture from this image.
        source_img = Image.open(source_path).convert("RGB").resize((384, 512))
        source_img = np.array(source_img).astype(np.float32) / 127.5 - 1.0
        source_img = torch.from_numpy(source_img.transpose(2, 0, 1)) # [3, H, W]

        return {
            "pixel_values": image,
            "mask": mask,
            "pose_values": pose,
            "cloth_values": source_img # This is now Person B's photo
        }