import torch
import torch.nn as nn


class AutoEncoderSegmentation(nn.Module):
    def __init__(self, pretrained_encoder: torch.nn.Module, num_classes: int = 3) -> None:
        """
        Initialize the AutoEncoderSegmentation model.

        This model leverages a pretrained encoder for feature extraction and a custom decoder
        for generating segmentation maps.

        Parameters:
            pretrained_encoder (torch.nn.Module): A pretrained encoder module for feature extraction.
            num_classes (int): The number of segmentation classes. Default is 3.

        Returns:
            None
        """
        super(AutoEncoderSegmentation, self).__init__()

        # Use the pretrained encoder for feature extraction.
        self.encoder = pretrained_encoder

        # Define the segmentation decoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Instead of a sigmoid activation, a Conv2d layer outputs logits for each class per pixel.
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, num_classes, H, W) containing segmentation logits.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
