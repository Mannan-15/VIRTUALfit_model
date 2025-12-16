import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from tqdm import tqdm

# Import our custom modules
from model import get_student_unet, CADM
from dataset import VITONDataset
from pipeline import LiteMPVTONPipeline

# --- CONFIGURATION ---
TEACHER_MODEL_ID = "runwayml/stable-diffusion-inpainting"
OUTPUT_DIR = "models/litemp_vton_fp16"
DATA_DIR = "data/viton_hd" 
BATCH_SIZE = 1  # Keep 1 for RTX 4050
NUM_EPOCHS = 10

def train():
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    print(f"--- Starting Training on {device} ---")

    # 1. Load Data
    train_dataset = VITONDataset(data_root=DATA_DIR, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Load Teacher (8-bit Quantized)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    teacher_unet = UNet2DConditionModel.from_pretrained(
        TEACHER_MODEL_ID, subfolder="unet", quantization_config=bnb_config
    )
    teacher_unet.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(TEACHER_MODEL_ID, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    noise_scheduler = DDPMScheduler.from_pretrained(TEACHER_MODEL_ID, subfolder="scheduler")

    # 3. Initialize Student & CADM
    student_unet = get_student_unet().to(device)
    cadm = CADM(teacher_channels=1280, student_channels=640).to(device)

    # --- AUTO-RESUME LOGIC ---
    # Check if we have a saved checkpoint to resume from
    latest_path = os.path.join(OUTPUT_DIR, "latest")
    start_epoch = 0
    
    if os.path.exists(latest_path):
        print(f"🔄 Resuming from checkpoint: {latest_path}")
        # Load weights
        student_unet = UNet2DConditionModel.from_pretrained(latest_path).to(device)
        # Try to guess epoch from folder names (optional simple logic)
        existing_checkpoints = [d for d in os.listdir(OUTPUT_DIR) if "epoch" in d]
        if existing_checkpoints:
            # sort by epoch number
            existing_checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            start_epoch = int(existing_checkpoints[-1].split('-')[-1]) + 1
            print(f"Resuming at Epoch {start_epoch+1}")
    else:
        print("🆕 Starting fresh training")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        list(student_unet.parameters()) + list(cadm.parameters()), 
        lr=1e-5
    )

    # 5. Pipeline Helper (For Real Poses)
    pose_extractor = LiteMPVTONPipeline(device=device)

    # Prepare
    student_unet, cadm, optimizer, train_dataloader = accelerator.prepare(
        student_unet, cadm, optimizer, train_dataloader
    )

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            student_unet.train()
            
            # A. Get Images & Mask
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            mask = batch["mask"].to(device, dtype=torch.float16)
            
            # VAE Encode
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            masked_image_latents = latents * (1 - mask)
            mask_64 = F.interpolate(mask, size=(64, 64), mode="nearest")

            # B. Get REAL Pose (Using Pipeline)
            # We need the original file path to feed DWPose
            img_filename = batch["filename"][0] # Batch size is 1
            img_path_full = os.path.join(DATA_DIR, "train", "image", img_filename)
            
            # Extract Pose -> Tensor -> Encode to Latents
            # Note: This is slow-ish. Ideally pre-compute this.
            with torch.no_grad():
                pose_tensor = pose_extractor.get_pose_tensor(img_path_full).to(device, dtype=torch.float16)
                pose_latents = vae.encode(pose_tensor).latent_dist.sample() * 0.18215

            # C. Add Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # D. Forward Passes
            
            # Teacher (Standard Input)
            teacher_input = torch.cat([noisy_latents, mask_64, masked_image_latents], dim=1)
            with torch.no_grad():
                t_out = teacher_unet(teacher_input, timesteps, return_dict=False, output_hidden_states=True)
                t_noise = t_out[0]
                t_feat = t_out[1][-1]

            # Student (Enhanced Input with Pose)
            student_input = torch.cat([noisy_latents, mask_64, masked_image_latents, pose_latents], dim=1)
            s_out = student_unet(student_input, timesteps, return_dict=False, output_hidden_states=True)
            s_noise = s_out[0]
            s_feat = s_out[1][-1]

            # E. Loss
            loss_mse = F.mse_loss(s_noise, noise)
            loss_distill = F.mse_loss(s_noise, t_noise)
            loss_cadm = cadm(t_feat, s_feat)

            loss = loss_mse + 0.5 * loss_distill + 0.03 * loss_cadm

            # Backward
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        # --- SAVE CHECKPOINT AFTER EVERY EPOCH ---
        print(f"💾 Saving Epoch {epoch+1} Checkpoint...")
        epoch_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        latest_dir = os.path.join(OUTPUT_DIR, "latest")
        
        # Save to specific epoch folder
        student_unet.save_pretrained(epoch_dir)
        # Update 'latest' folder (for easy resuming)
        student_unet.save_pretrained(latest_dir)

if __name__ == "__main__":
    train()
    