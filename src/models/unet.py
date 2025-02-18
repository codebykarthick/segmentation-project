import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    """ UNet Encoder Model for image segmentation based on the original paper. """

    def __init__(self, in_channels=3, base_channels=64):
        super(UNetEncoder, self).__init__()

        # Encoder Blocks with two convolution layers each
        self.enc1 = self.conv_block(
            in_channels, base_channels)             # (H, W, 64)
        self.enc2 = self.conv_block(
            base_channels, base_channels * 2)       # (H/2, W/2, 128)
        self.enc3 = self.conv_block(
            base_channels * 2, base_channels * 4)   # (H/4, W/4, 256)
        self.enc4 = self.conv_block(
            base_channels * 4, base_channels * 8)   # (H/8, W/8, 512)
        self.bottleneck = self.conv_block(
            base_channels * 8, base_channels * 16)  # (H/16, W/16, 1024)

        # Max-pooling layers for downsampling
        self.pool = nn.MaxPool2d(2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """ Basic convolutional block: Conv → ReLU → Conv → ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """ The forward pass of the encoder. """
        skip_connections = []  # Save the skip connections for the decoder

        # First block
        x = self.enc1(x)
        skip_connections.append(x)
        x = self.pool(x)

        # Second block
        x = self.enc2(x)
        skip_connections.append(x)
        x = self.pool(x)

        # Third block
        x = self.enc3(x)
        skip_connections.append(x)
        x = self.pool(x)

        # Fourth block
        x = self.enc4(x)
        skip_connections.append(x)
        x = self.pool(x)

        # Bottleneck block
        x = self.bottleneck(x)

        return x, skip_connections


class UNetDecoder(nn.Module):
    """ UNet Decoder Model for image segmentation based on the original paper. """

    def __init__(self, out_channels=3, base_channels=64):
        super(UNetDecoder, self).__init__()

        # Upsampling and conv block 1
        self.upconv1 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(
            base_channels * 16, base_channels * 8)

        # Block 2
        self.upconv2 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(
            base_channels * 8, base_channels * 4)

        # Block 3
        self.upconv3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(
            base_channels * 4, base_channels * 2)

        # Block 4
        self.upconv4 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(
            base_channels * 2, base_channels)

        # Segmentation Layer
        self.segmentation = nn.Conv2d(
            base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """ Basic convolutional block: Conv → ReLU → Conv → ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connections):
        """Decoder forward pass with skip connections."""
        x = self.upconv1(x)
        x = torch.cat((x, skip_connections[3]), dim=1)
        x = self.dec1(x)

        x = self.upconv2(x)
        x = torch.cat((x, skip_connections[2]), dim=1)
        x = self.dec2(x)

        x = self.upconv3(x)
        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.dec3(x)

        x = self.upconv4(x)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.dec4(x)

        x = self.segmentation(x)

        return x


class UNet(nn.Module):
    """Complete UNet Model combining encoder and decoder"""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.decoder = UNetDecoder(out_channels, base_channels)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x
