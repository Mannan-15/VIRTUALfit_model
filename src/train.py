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

# --- CONFIGURATION ---
DATA_DIR = "/kaggle/input/viton-hd-preprocessed/viton_hd"
OUTPUT_DIR = "/kaggle/working/models"

# PATH TO YOUR EPOCH 7 CHECKPOINT (Update this path!)
PREVIOUS_CHECKPOINT = "/kaggle/input/virtual-fit-checkpoints/checkpoint-epoch-7" 

BATCH_SIZE = 4
NUM_EPOCHS = 5  # Refinement run

def load_and_patch_model(model_path, device):
    """
    Loads a 13-channel model and grafts it onto a 17-channel model.
    The 4 new 'cloth' channels are initialized to zero.
    """
    print(f"🔧 Patching model from {model_path}...")
    
    # 1. Initialize a clean 17-channel UNet (Random weights)
    # Inputs: Noisy(4) + Mask(1) + MaskedImg(4) + Pose(4) + Cloth(4) = 17
    new_model = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        subfolder="unet", 
        in_channels=17, 
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    ).to(device)

    # 2. Load the trained 13-channel weights
    try:
        old_model = UNet2DConditionModel.from_pretrained(model_path)
        old_state_dict = old_model.state_dict()
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return new_model

    # 3. Graft weights
    new_state_dict = new_model.state_dict()
    
    for key in new_state_dict:
        # If it's a standard layer present in both, copy it
        if key in old_state_dict and new_state_dict[key].shape == old_state_dict[key].shape:
            new_state_dict[key] = old_state_dict[key]
            
        # If it's the input convolution (where sizes differ)
        elif "conv_in.weight" in key:
            print("   -> Adjusting input layer dimensions...")
            old_weight = old_state_dict[key] # [320, 13, 3, 3]
            new_weight = new_state_dict[key] # [320, 17, 3, 3]
            
            # Copy trained weights (Channels 0-12)
            new_weight[:, :13, :, :] = old_weight
            
            # Zero-initialize new Cloth weights (Channels 13-16)
            # This ensures the model output doesn't jump wildly at step 0
            new_weight[:, 13:, :, :] = torch.zeros_like(new_weight[:, 13:, :, :])
            
            new_state_dict[key] = new_weight

    # 4. Load patched dictionary
    new_model.load_state_dict(new_state_dict)
    print("✅ Model successfully patched to 17 channels!")
    return new_model

def train():
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"--- Starting V2 (Cloth-Aware) Training on {device} ---")

    # 1. Load Data
    train_dataset = VITONDataset(data_root=DATA_DIR, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Load Helpers
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False)
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
    
    # Load Teacher (Standard 9-channel, frozen)
    teacher_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    teacher_unet.requires_grad_(False)

    # 3. LOAD & PATCH STUDENT MODEL
    student_unet = load_and_patch_model(PREVIOUS_CHECKPOINT, device)
    
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=1e-5)
    student_unet, optimizer, train_dataloader = accelerator.prepare(student_unet, optimizer, train_dataloader)

    # Text Embeddings (Empty)
    with torch.no_grad():
        empty_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        base_embeddings = text_encoder(empty_input.input_ids.to(device))[0]

    # --- TRAINING LOOP ---
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            student_unet.train()
            
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            mask = batch["mask"].to(device, dtype=torch.float16)
            pose_values = batch["pose_values"].to(device, dtype=torch.float16)
            cloth_values = batch["cloth_values"].to(device, dtype=torch.float16) # NEW

            # Repeat embeddings for batch
            batch_embeddings = base_embeddings.repeat(pixel_values.shape[0], 1, 1)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                pose_latents = vae.encode(pose_values).latent_dist.sample() * 0.18215
                cloth_latents = vae.encode(cloth_values).latent_dist.sample() * 0.18215 # NEW

            mask_64 = F.interpolate(mask, size=(64, 64), mode="nearest")
            masked_image_latents = latents * (1 - mask_64)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                # Teacher (Standard Input: 9 channels)
                teacher_input = torch.cat([noisy_latents, mask_64, masked_image_latents], dim=1)
                with torch.no_grad():
                    t_noise = teacher_unet(teacher_input, timesteps, encoder_hidden_states=batch_embeddings, return_dict=False)[0]

                # Student (New V2 Input: 17 channels)
                student_input = torch.cat([
                    noisy_latents,          # 4
                    mask_64,                # 1
                    masked_image_latents,   # 4
                    pose_latents,           # 4
                    cloth_latents           # 4 (New)
                ], dim=1)
                
                s_noise = student_unet(student_input, timesteps, encoder_hidden_states=batch_embeddings, return_dict=False)[0]

                loss = F.mse_loss(s_noise, noise) + 0.5 * F.mse_loss(s_noise, t_noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        # Save
        print(f"💾 Saving Epoch {epoch+1}...")
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-v2-epoch-{epoch+1}")
        student_unet.save_pretrained(save_path)
        student_unet.save_pretrained(os.path.join(OUTPUT_DIR, "latest"))

if __name__ == "__main__":
    train()