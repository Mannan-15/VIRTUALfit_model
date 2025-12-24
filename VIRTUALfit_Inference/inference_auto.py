import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, SegformerImageProcessor, AutoModelForSemanticSegmentation
from controlnet_aux import OpenposeDetector
from tqdm.auto import tqdm
import torch.nn.functional as F

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model" # Your V2 Checkpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True, help="Path to User Image")
    parser.add_argument("--cloth", type=str, required=True, help="Path to Reference Person/Cloth Image")
    parser.add_argument("--output", type=str, default="final_tryon.png")
    return parser.parse_args()

# --- 1. AUTOMATION MODELS ---
print("⏳ Loading Helper Models...")
# Pose
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(DEVICE)
# Masking
seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE)

def process_inputs(user_path, cloth_path):
    print("🤖 Automating Mask & Pose...")
    user_img = Image.open(user_path).convert("RGB").resize((384, 512))
    cloth_img = Image.open(cloth_path).convert("RGB").resize((384, 512))

    # A. Generate Pose
    pose_img = pose_detector(user_img, include_body=True, include_hand=False, include_face=False)
    pose_img = pose_img.resize((384, 512))

    # B. Generate Mask (Upper Body + Torso Skin)
    inputs = seg_processor(images=user_img, return_tensors="pt").to(DEVICE)
    outputs = seg_model(**inputs)
    logits = outputs.logits
    upsampled_logits = F.interpolate(logits, size=(512, 384), mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # Mask Logic: 4=Upper, 5=Dress, 6=Coat, 7=TorsoSkin (Check model card for exact indices)
    # Note: mattmdjaga/segformer_b2_clothes indices: 4: Upper-clothes, 5: Dress, 6: Coat, 7: Socks... 
    # Let's target UpperBody(4) + Dress(5) + Coat(6)
    mask = torch.zeros_like(pred_seg, dtype=torch.float32)
    mask[pred_seg == 4] = 1.0 
    mask[pred_seg == 5] = 1.0 
    mask[pred_seg == 6] = 1.0
    
    # Convert to Tensor Inputs for UNet
    def to_tensor(img):
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float16)

    pixel_values = to_tensor(user_img)
    pose_values = to_tensor(pose_img)
    cloth_values = to_tensor(cloth_img) # The Reference Person
    
    # Mask Tensor
    mask_tensor = mask.unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=torch.float16) # [1,1,H,W]

    return pixel_values, mask_tensor, pose_values, cloth_values

# --- 2. MAIN INFERENCE ---
@torch.no_grad()
def main():
    args = get_args()
    
    # Load VTON Model
    print(f"Loading VTON Model from {MODEL_PATH}...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    
    # Load Standard SD Components
    base = "runwayml/stable-diffusion-inpainting"
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float16).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # Get Inputs
    pixel_values, mask, pose_values, cloth_values = process_inputs(args.user, args.cloth)

    # Encode Latents
    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
    pose_latents = vae.encode(pose_values).latent_dist.sample() * 0.18215
    cloth_latents = vae.encode(cloth_values).latent_dist.sample() * 0.18215
    
    mask_64 = F.interpolate(mask, size=(64, 48), mode="nearest")
    masked_image_latents = latents * (1 - mask_64)

    # Empty Text
    empty_token = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    encoder_hidden_states = text_encoder(empty_token.input_ids.to(DEVICE))[0]

    # Denoise
    scheduler.set_timesteps(30)
    current_latents = torch.randn_like(latents)
    
    print("✨ Generating Try-On...")
    for t in tqdm(scheduler.timesteps):
        # 17 Channels
        model_input = torch.cat([current_latents, mask_64, masked_image_latents, pose_latents, cloth_latents], dim=1)
        
        noise_pred = unet(model_input, t, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        current_latents = scheduler.step(noise_pred, t, current_latents).prev_sample

    # Decode
    current_latents = current_latents / 0.18215
    image = vae.decode(current_latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    
    Image.fromarray(image).save(args.output)
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
    