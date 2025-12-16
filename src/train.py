import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from tqdm import tqdm

# Import our custom modules
from model import get_student_unet
from dataset import VITONDataset
# REMOVED: from pipeline import LiteMPVTONPipeline (Not needed for Turbo Mode)

# --- CONFIGURATION ---
TEACHER_MODEL_ID = "runwayml/stable-diffusion-inpainting"
OUTPUT_DIR = "models/litemp_vton_fp16"
DATA_DIR = "data/viton_hd" 
BATCH_SIZE = 1
NUM_EPOCHS = 10

def train():
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    print(f"--- Starting Turbo Training on {device} ---")

    # 1. Load Data
    train_dataset = VITONDataset(data_root=DATA_DIR, mode="train")
    # CHANGE: num_workers=4 is now safe and much faster
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Load Teacher (Frozen, 8-bit)
    print("Loading Teacher...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    teacher_unet = UNet2DConditionModel.from_pretrained(
        TEACHER_MODEL_ID, subfolder="unet", quantization_config=bnb_config
    )
    teacher_unet.requires_grad_(False)

    # Load VAE (Force float16)
    vae = AutoencoderKL.from_pretrained(
        TEACHER_MODEL_ID, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    
    # Load Text Encoder
    text_encoder = CLIPTextModel.from_pretrained(TEACHER_MODEL_ID, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(TEACHER_MODEL_ID, subfolder="tokenizer")
    text_encoder.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(TEACHER_MODEL_ID, subfolder="scheduler")

    # 3. Initialize Student
    print("Initializing Student...")
    student_unet = get_student_unet().to(device)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=1e-5)

    # REMOVED: pose_extractor initialization (Not needed)

    # Prepare with Accelerator
    student_unet, optimizer, train_dataloader = accelerator.prepare(
        student_unet, optimizer, train_dataloader
    )

    # --- PRE-COMPUTE EMPTY TEXT EMBEDDINGS ---
    with torch.no_grad():
        empty_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        empty_embeddings = text_encoder(empty_input.input_ids.to(device))[0].to(dtype=torch.float16)

    # --- TRAINING LOOP ---
    start_epoch = 0
    if os.path.exists(os.path.join(OUTPUT_DIR, "latest")):
        print("🔄 Resuming from latest checkpoint...")
        try:
            student_unet = UNet2DConditionModel.from_pretrained(os.path.join(OUTPUT_DIR, "latest")).to(device)
        except:
            print("Warning: Could not load checkpoint, starting fresh.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            student_unet.train()

            # A. Prepare Inputs
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            mask = batch["mask"].to(device, dtype=torch.float16)
            
            # --- NEW: Load Pre-Computed Pose Directly ---
            pose_values = batch["pose_values"].to(device, dtype=torch.float16)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                # Encode Pose to Latents
                pose_latents = vae.encode(pose_values).latent_dist.sample() * 0.18215

            mask_64 = F.interpolate(mask, size=(64, 64), mode="nearest")
            masked_image_latents = latents * (1 - mask_64)

            # B. Add Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # C. Forward Passes (AutoCast for Speed)
            with accelerator.autocast():
                # Teacher
                teacher_input = torch.cat([noisy_latents, mask_64, masked_image_latents], dim=1)
                with torch.no_grad():
                    t_noise = teacher_unet(
                        teacher_input, timesteps, 
                        encoder_hidden_states=empty_embeddings, return_dict=False
                    )[0]

                # Student
                student_input = torch.cat([noisy_latents, mask_64, masked_image_latents, pose_latents], dim=1)
                s_noise = student_unet(
                    student_input, timesteps, 
                    encoder_hidden_states=empty_embeddings, return_dict=False
                )[0]

                loss_mse = F.mse_loss(s_noise, noise)
                loss_distill = F.mse_loss(s_noise, t_noise)

            # D. Backward
            loss = (loss_mse + 0.5 * loss_distill).float()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        # Save Checkpoint
        print(f"💾 Saving Epoch {epoch+1}...")
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        student_unet.save_pretrained(save_path)
        student_unet.save_pretrained(os.path.join(OUTPUT_DIR, "latest"))

if __name__ == "__main__":
    train()