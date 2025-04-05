import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ClipSegmentation(nn.Module):
    """
    A segmentation network that:
      1) Uses CLIP (ViT-B/32) as a frozen image encoder to get a 16×16 spatial feature map.
      2) Learns a decoder to upsample from 16×16 to 512×512 and predict a multi-class mask.
    """

    def __init__(self, device="cpu", num_classes=3, clip_model_name="ViT-B/32"):
        print(f"Using device: {device}")
        super().__init__()

        # Load CLIP and freeze its parameters
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Example decoder: 5 "blocks" of transpose-convolution to go from 16×16 → 512×512
        # Each block roughly doubles the spatial dimension. Adjust channel sizes as needed.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Finally, predict the segmentation mask with num_classes channels
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Expects x in shape (B, 3, H, W) with H=W=512.
        Returns segmentation logits of shape (B, num_classes, 512, 512).
        """

        # Make sure input is 512×512 for the CLIP ViT-B/32:
        # CLIP’s ViT-B/32 uses a 32×32 patch size, so 512×512 → 16×16 patches + 1 CLS token.
        # If your images are not already 512×512, resize them here:
        if x.shape[-1] != 512:
            x = F.interpolate(x, size=(512, 512),
                              mode='bilinear', align_corners=False)

        # Move to same device/dtype as CLIP
        device = next(self.clip_model.parameters()).device
        x = x.to(device=device, dtype=self.clip_model.dtype)

        # -- 1) Pass through CLIP’s visual stem manually to get the final patch embeddings --
        # clip_model.visual includes the patch-embedding conv, transformer, etc.
        # By default, clip_model.encode_image(...) returns a pooled feature, but for segmentation
        # we want the spatial feature map from the transformer.
        with torch.no_grad():
            # Step 1: patch + positional embedding
            # (B, 3, 512, 512) -> (B, 768, 16, 16) after conv1
            x = self.clip_model.visual.conv1(x)  # shape: [B, 768, 16, 16]
            # shape: [B, 16*16, 768] = [B, 256, 768]
            x = x.flatten(2).transpose(1, 2)
            x = self.clip_model.visual.ln_pre(x)  # layernorm

            # Step 2: Transformer
            # shape: [B, 257, 768] (includes class token)
            x = self.clip_model.visual.transformer(x)
            x = self.clip_model.visual.ln_post(x)

            # Step 3: If a class token is present (i.e. output has 257 tokens), drop it. Otherwise, keep all tokens.
            if x.shape[1] == 257:
                x = x[:, 1:, :]

        # Reshape to 2D feature map: (B, 768, 16, 16)
        B, N, C = x.shape  # N = 16*16 = 256
        h = w = int(N**0.5)  # 16
        x = x.transpose(1, 2).view(B, C, h, w)
        x = x.to(torch.float32)

        # -- 2) Decode the 16×16 features up to 512×512 --
        x = self.decoder(x)  # shape: (B, num_classes, 512, 512)

        return x
