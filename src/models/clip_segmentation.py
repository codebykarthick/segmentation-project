import torch
import torch.nn as nn
import clip
import torch.nn.functional as F


class ClipSegmentation(nn.Module):
    """
    Use a pretrained CLIP model as the image encoder, and add a segmentation
    head on top of the final embeddings.

    Attributes:
        clip_model: The pretrained CLIP model used for image encoding.
        embed_to_features: A sequential module transforming the CLIP embedding.
        post_clip_conv: A sequential module acting as a segmentation decoder.
        initial_h (int): Initial height of the spatial grid.
        initial_w (int): Initial width of the spatial grid.
    """

    def __init__(self, clip_model_name: str = "ViT-B/32", num_classes: int = 3, device: str = "cpu") -> None:
        """
        Initializes the ClipSegmentation model.

        Args:
            clip_model_name (str): The name of the CLIP model to load. Defaults to "ViT-B/32".
            num_classes (int): Number of segmentation classes. Defaults to 3.
            device (str): Device on which to load the model (e.g., "cpu", "cuda"). Defaults to "cpu".

        Returns:
            None
        """
        super(ClipSegmentation, self).__init__()

        # 1) Load CLIP
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 2) CLIP’s visual encoder outputs an embedding of shape [batch_size, channel_dim].
        #    For ViT-B/32, channel_dim is 512. We'll treat these as "global" embeddings.
        #    For segmentation, we typically want a spatial map, but for a simple baseline,
        #    we can either:
        #       (A) reshape tokens from earlier layers, or
        #       (B) upsample from the global embedding.
        #    Here, we do a naive approach that broadcasts the global embedding into a
        #    low-res HxW, then upsamples.

        embed_dim = self.clip_model.visual.output_dim  # e.g. 512 for ViT-B/32

        # 3) Optional: simple 1-layer to transform the 512-dim embedding
        self.embed_to_features = nn.Sequential(
            nn.Linear(embed_dim, 512),   # map 512 -> 512
            nn.ReLU(inplace=True),
        )

        # 4) A naive approach: expand that 512-dimensional embedding to a small 2D grid
        #    (say 8x8), then do conv-transpose or a typical "decoder" to get back to original image size.
        #    This is just a demonstration. Feel free to adapt for your own resolution.

        # Let’s fix a small 8x8 spatial shape:
        self.initial_h = 8
        self.initial_w = 8

        # You could store a "template" that we fill with repeated embeddings
        # shape => [batch_size, 512, 8, 8]
        self.post_clip_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
        # The output has shape [batch_size, num_classes, <final_H>, <final_W>].
        # If your training images are bigger than that final size, you can do another
        # upsampling or more transposed convolution layers.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input images with shape [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Segmentation logits with shape [batch_size, num_classes, H', W'].
        """
        # 1) Preprocess images to the size CLIP expects (224x224 if using ViT-B/32)
        #    or rely on transforms. For a quick hack:
        #    If x is not 224x224, we can do something like:
        original_h, original_w = x.shape[2], x.shape[3]  # e.g. 560, 600
        x_small = F.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False)

        # 2) Extract CLIP embeddings
        with torch.no_grad():
            # shape: [batch_size, 512] for ViT-B/32
            clip_emb = self.clip_model.encode_image(x_small)

        # 3) Map the 512 embeddings -> a new 512 dimension (optional)
        #    shape => [batch_size, 512]
        feat = self.embed_to_features(clip_emb)

        # 4) Expand that [batch_size, 512] to [batch_size, 512, 1, 1]
        feat = feat.unsqueeze(-1).unsqueeze(-1)

        # 5) Then tile or otherwise expand up to [batch_size, 512, 8, 8]
        feat = feat.expand(-1, 512, self.initial_h, self.initial_w)

        # 6) Pass through the segmentation decoder
        out = self.post_clip_conv(feat)

        return out
