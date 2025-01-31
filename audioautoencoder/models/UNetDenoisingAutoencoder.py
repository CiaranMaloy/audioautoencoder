import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.output_layer = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Final output layer

    def forward(self, x, verbose=True):
        if verbose:
            print('interpolating', x.shape)
        # oversample input
        x = F.interpolate(x, size=(1028, 182), mode='bilinear', align_corners=False)
        if verbose:
            print('input', x.shape)

        # Encoder
        e1 = self.encoder1(x)  # Level 1
        if verbose:
            print('e1', e1.shape)

        e2 = self.encoder2(e1)  # Level 2
        if verbose:
            print('e2', e2.shape)

        e3 = self.encoder3(e2)  # Level 3
        if verbose:
            print('e3', e3.shape)

        # Bottleneck
        b = self.bottleneck(e3)
        if verbose:
            print('b', b.shape)

        # Decoder
        b = F.pad(b, (0, 0, 0, 0))
        e3 = F.pad(e3, (0, 0, 0, 0))
        d3 = self.decoder3(torch.cat((b, e3), dim=1))  # Concatenate bottleneck and e3
        if verbose:
            print('d3', d3.shape)

        d3 = F.pad(d3, (0, 0, 0, 0))
        e2 = F.pad(e2, (0, 0, 0, 0))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # Concatenate d3 and e2
        if verbose:
            print('d2', d2.shape)

        d2 = F.pad(d2, (0, 0, 0, 0))
        e1 = F.pad(e1, (0, 0, 0, 0))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # Concatenate d2 and e1
        if verbose:
            print('d1', d1.shape)

        raw_output = self.output_layer(d1)  # Final output
        if verbose:
            print('output', output.shape)
        output = F.interpolate(raw_output, size=(1025, 175), mode='bilinear', align_corners=False)
        if verbose:
            print('output post interpolation', output.shape)
        return output
