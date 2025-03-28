import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """ Autoencoder model for image compression and reconstruction """

    def __init__(self) -> None:
        """
        Initialize the Autoencoder model.

        This model compresses the input image and then reconstructs it using an encoder-decoder architecture.

        Returns:
            None
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # Downsample (H/2, W/2)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)  # Final compression (H/4, W/4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the autoencoder.

        Parameters:
            x (torch.Tensor): Input tensor with shape (N, 3, H, W).

        Returns:
            torch.Tensor: Reconstructed output tensor with shape (N, 3, H, W).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
