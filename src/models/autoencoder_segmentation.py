import torch
import torch.nn as nn


class AutoEncoderSegmentation(nn.Module):
    def __init__(self, pretrained_encoder, num_classes=3):
        super(AutoEncoderSegmentation, self).__init__()

        # Use the pretrained encoder
        self.encoder = pretrained_encoder

        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # This is the only difference from the Autoencoder's decoder.
            # Instead of a sigmoid activation, we use a Conv2d layer to output
            # the number of classes, for each pixel, denoting the class.
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
