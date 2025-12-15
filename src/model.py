import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

class CADM(nn.Module):
    """
    Cross-Attention Alignment Distillation Module (Equation 5 from paper).
    Aligns Student features (Key/Value) to Teacher features (Query).
    """
    def __init__(self, teacher_channels, student_channels):
        super().__init__()
        # Projections to map features to shared dimension if needed
        self.query_proj = nn.Conv2d(teacher_channels, teacher_channels, 1)
        self.key_proj = nn.Conv2d(student_channels, teacher_channels, 1)
        self.value_proj = nn.Conv2d(student_channels, teacher_channels, 1)
        self.scale = teacher_channels ** -0.5

    def forward(self, teacher_feat, student_feat):
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        B, C_t, H, W = teacher_feat.shape

        Q = self.query_proj(teacher_feat).view(B, C_t, -1).permute(0, 2, 1) # (B, N, C)
        K = self.key_proj(student_feat).view(B, C_t, -1)                    # (B, C, N)
        V = self.value_proj(student_feat).view(B, C_t, -1).permute(0, 2, 1) # (B, N, C)

        # Cross Attention: Q(Teacher) searching in K(Student)
        attn_scores = torch.bmm(Q, K) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Reconstruct teacher-like features from Student info
        aligned_feat = torch.bmm(attn_probs, V) # (B, N, C)

        # Reshape back to image
        aligned_feat = aligned_feat.permute(0, 2, 1).view(B, C_t, H, W)

        # Loss: The aligned student features should match the Teacher features
        # Using L1 Loss as per Eq 5 context implies distance minimization
        loss = F.l1_loss(aligned_feat, teacher_feat)
        return loss

def get_student_unet():
    """
    Creates the LiteMP-VTON Student UNet (~286M Params).
    Config adapted from paper: Channels [320, 448, 640, 640]
    """
    # Load config from standard SD 1.5 Inpainting
    try:
        config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-inpainting", subfolder="unet")
    except:
        # Fallback if config isn't cached yet
        config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-inpainting", subfolder="unet", force_download=True)

    # 1. Modify Channels for Compression (Paper Section 3.4.2)
    # Original: [320, 640, 1280, 1280] -> Student: [320, 448, 640, 640]
    config['block_out_channels'] = (320, 448, 640, 640)

    # 2. Update Input Channels to 13 
    # (4 Latent + 4 Masked Latent + 1 Mask + 4 Pose Latent)
    config['in_channels'] = 13 

    # 3. Create Model
    student_unet = UNet2DConditionModel.from_config(config)

    # 4. Expand Input Weights for 13 Channels
    # Initialize new channels (Pose) to zero to not break pretrained weights initially
    with torch.no_grad():
        old_conv = student_unet.conv_in
        new_conv = nn.Conv2d(13, old_conv.out_channels, kernel_size=3, padding=1)

        # Copy original 9 channel weights (Latent+Masked+Mask)
        # We initialize with random weights here because we are training from scratch (Distillation)
        # but copying gives a slightly better starting point for the standard channels.
        if old_conv.weight.shape[1] == 9:
             new_conv.weight[:, :9, :, :] = old_conv.weight
        else:
             # If config loaded default 4-channel SD, we expand to 9 then 13
             # But since we loaded inpainting config, it should be 9.
             pass

        # Zero out the new 4 pose channels so they don't add noise at step 0
        new_conv.weight[:, 9:, :, :] = 0
        new_conv.bias = old_conv.bias

        student_unet.conv_in = new_conv

    return student_unet