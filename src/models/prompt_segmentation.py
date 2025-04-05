import torch
import torch.nn as nn
from .unet import UNetEncoder, UNetDecoder


class PromptSegmentation(nn.Module):
    """
    UNet-based model for prompt-based segmentation.
    Takes image + binary prompt mask as 4-channel input.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64) -> None:
        """
        Initializes the PromptSegmentation model using UNet architecture.

        Args:
            in_channels (int): Number of channels in the input (image + prompt). Defaults to 4.
            out_channels (int): Number of output segmentation classes. Defaults to 3.
            base_channels (int): Base number of filters for the convolutional layers. Defaults to 64.
        """
        super(PromptSegmentation, self).__init__()
        self.encoder = UNetEncoder(
            in_channels=in_channels, base_channels=base_channels)
        self.decoder = UNetDecoder(
            out_channels=out_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the prompt-based segmentation model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 4, H, W]

        Returns:
            torch.Tensor: Output segmentation mask of shape [batch_size, num_classes, H, W]
        """
        original_h, original_w = x.shape[2], x.shape[3]
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections, (original_h, original_w))
        return x
