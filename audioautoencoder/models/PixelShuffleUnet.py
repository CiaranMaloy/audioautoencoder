import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffleUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[32, 64, 128]):
        super(PixelShuffleUNet, self).__init__()
        
        # Encoder layers
        self.encoder1 = self.conv_block(in_channels, features[0])
        self.encoder2 = self.conv_block(features[0], features[1])
        self.encoder3 = self.conv_block(features[1], features[2])

        # Bottleneck
        self.bottleneck = self.conv_block(features[2], features[2] * 2)

        # Decoder layers with PixelShuffle upsampling
        self.decoder3 = self.upsample_block(features[2] * 2, features[2])
        self.decoder2 = self.upsample_block(features[2], features[1])
        self.decoder1 = self.upsample_block(features[1], features[0])

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """Standard Conv -> BatchNorm -> ReLU block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upsample_block(self, in_channels, out_channels, upscale_factor=2):
        """Upsampling using PixelShuffle"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2, stride=2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, kernel_size=2, stride=2))

        # Decoder with skip connections
        d3 = F.pad(self.decoder3(b), (0, 1, 0, 0)) + e3
        d2 = F.pad(self.decoder2(d3), (0, 1, 0, 0)) + e2
        d1 = F.pad(self.decoder1(d2), (0, 1, 0, 1)) + e1

        return self.final_conv(d1)

# Test with random input
if __name__ == "__main__":
    model = PixelShuffleUNet(in_channels=2, out_channels=2)
    x = torch.randn((8, 2, 1025, 175))  # Batch size 1, 128x128 input
    out = model(x)
    print(out.shape)  # Expected: (1, 1, 128, 128)