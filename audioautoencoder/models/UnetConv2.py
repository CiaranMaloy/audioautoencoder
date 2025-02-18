import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConv2(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNetConv2, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 64, 7)
        self.enc2 = self.conv_block(64, 128, 5)
        self.enc3 = self.conv_block(128, 256, 3)
        self.enc4 = self.conv_block(256, 512, 3)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, 1)

        # Decoder (Upsampling)
        self.dec4 = self.upconv_block(1024, 512, 3)
        self.dec3 = self.upconv_block(512+512, 256, 3)
        self.dec2 = self.upconv_block(256+256, 128, 5)
        self.dec1 = self.upconv_block(128+128, 64, 7)

        # Final Output Layer
        self.final = nn.Conv2d(64+64, out_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels, kernel_size, dropout=0.2):
        """Convolutional Block with Dropout in Deeper Layers Only"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Dropout only for deeper encoder layers
        if out_channels >= 256:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, kernel_size, dropout=0.2):
        """Upsampling Block with Dropout in First Few Decoder Layers"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Dropout only for first few decoder layers
        if in_channels >= 1024:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)


    def forward(self, x):
        """Forward pass with skip connections"""
        # Encoding
        e1 = self.enc1(x)  # (batch, 64, 1028, 175)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))  # (batch, 128, 514, 87)
        e3 = self.enc3(nn.functional.avg_pool2d(e2, 2))  # (batch, 256, 257, 43)
        e4 = self.enc4(nn.functional.avg_pool2d(e3, 2))  # (batch, 512, 128, 21)

        # Bottleneck
        b = self.bottleneck(nn.functional.avg_pool2d(e4, 2))  # (batch, 1024, 64, 10)

        # Decoding + Skip Connections
        d4 = self.dec4(b)  # (batch, 512, ?, ?)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)  # (batch, 256, ?, ?)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)  # (batch, 128, ?, ?)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)  # (batch, 64, ?, ?)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)

        # Final Convolution (output denoised spectrogram)
        return F.interpolate(self.final(d1), size=(1025, 175), mode="bilinear", align_corners=False)

# Test model
if __name__ == "__main__":
    model = UNetConv2()
    sample_input = torch.randn(2, 2, 1025, 175)  # (batch_size, channels, height, width)
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should match input shape (2, 2, 1028, 175)
