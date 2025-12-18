import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
from model import get_student_unet
from dataset import VITONDataset

# --- KAGGLE CONFIGURATION ---
DATA_DIR = "/kaggle/input/viton-hd-preprocessed/viton_hd" 
OUTPUT_DIR = "/kaggle/working/models" 

# Batch Size 4 is safe for Kaggle T4
BATCH_SIZE = 4
NUM_EPOCHS = 10

def train():
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"--- Starting Kaggle Training on {device} ---")

    # 1. Load Data
    train_dataset = VITONDataset(data_root=DATA_DIR, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Load Models
    print("Loading Teacher (FP16)...")
    teacher_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        subfolder="unet", 
        torch_dtype=torch.float16
    ).to(device)
    teacher_unet.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="tokenizer")
    text_encoder.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")

    print("Initializing Student...")
    student_unet = get_student_unet().to(device)
    
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=1e-5)

    student_unet, optimizer, train_dataloader = accelerator.prepare(
        student_unet, optimizer, train_dataloader
    )

    with torch.no_grad():
        empty_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        # Base embeddings (Batch Size 1)
        base_embeddings = text_encoder(empty_input.input_ids.to(device))[0]

    # --- TRAINING LOOP ---
    start_epoch = 0
    if os.path.exists(os.path.join(OUTPUT_DIR, "latest")):
        try:
            student_unet = UNet2DConditionModel.from_pretrained(os.path.join(OUTPUT_DIR, "latest")).to(device)
            print("Resumed from checkpoint.")
        except: pass

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            student_unet.train()
            
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            mask = batch["mask"].to(device, dtype=torch.float16)
            pose_values = batch["pose_values"].to(device, dtype=torch.float16)

            # --- FIX: REPEAT EMBEDDINGS TO MATCH BATCH SIZE ---
            # Get current batch size (it might be smaller than 4 at the very end of dataset)
            current_bs = pixel_values.shape[0]
            batch_embeddings = base_embeddings.repeat(current_bs, 1, 1)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                pose_latents = vae.encode(pose_values).latent_dist.sample() * 0.18215

            mask_64 = F.interpolate(mask, size=(64, 64), mode="nearest")
            masked_image_latents = latents * (1 - mask_64)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                # Teacher
                teacher_input = torch.cat([noisy_latents, mask_64, masked_image_latents], dim=1)
                with torch.no_grad():
                    t_noise = teacher_unet(
                        teacher_input, 
                        timesteps, 
                        encoder_hidden_states=batch_embeddings, # Using the repeated embeddings
                        return_dict=False
                    )[0]

                # Student
                student_input = torch.cat([noisy_latents, mask_64, masked_image_latents, pose_latents], dim=1)
                s_noise = student_unet(
                    student_input, 
                    timesteps, 
                    encoder_hidden_states=batch_embeddings, # Using the repeated embeddings
                    return_dict=False
                )[0]

                loss = F.mse_loss(s_noise, noise) + 0.5 * F.mse_loss(s_noise, t_noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"💾 Saving Epoch {epoch+1}...")
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        student_unet.save_pretrained(save_path)
        student_unet.save_pretrained(os.path.join(OUTPUT_DIR, "latest"))

if __name__ == "__main__":
    train()