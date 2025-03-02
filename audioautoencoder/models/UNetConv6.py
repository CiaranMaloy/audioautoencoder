import torch
import torch.nn as nn
import torch.nn.functional as F

# too big without.... lets try, halfing the expansion
# apternaitvely use max pooling instead of average pooling
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Avg Pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max Pooling
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))  # Convolution
        return x * self.sigmoid(attn)  # Apply Attention Map

class UNetConv6(nn.Module):
    # Update from UnetConv3, stride dimentionality reduction + attention on skip connections
    def __init__(self, in_channels=9, out_channels=4):
        super(UNetConv6, self).__init__()

        a = 2
        A, B, C, D = 64, 128, 256, 512
        bottleneck_channels = 1024

        # Encoder (Downsampling)
        enc_channels = [in_channels, A, B, C, D]
        self.enc1 = self.conv_block(enc_channels[0], enc_channels[1], 7, 1)
        self.enc2 = self.conv_block(enc_channels[1], enc_channels[2], 5, 2)
        self.enc3 = self.conv_block(enc_channels[2], enc_channels[3], 3, 2)
        self.enc4 = self.conv_block(enc_channels[3], enc_channels[4], 3, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(enc_channels[4], bottleneck_channels, 3, 2)

        # Decoder (Upsampling)
        dec_channels = [bottleneck_channels, D, C, B, A]
        self.dec4 = self.upconv_block(dec_channels[0], dec_channels[1], 3, 2)
        self.dec3 = self.upconv_block(dec_channels[1] + enc_channels[4], dec_channels[2], 3, 2)
        self.dec2 = self.upconv_block(dec_channels[2] + enc_channels[3], dec_channels[3], 5, 2)
        self.dec1 = self.upconv_block(dec_channels[3] + enc_channels[2], dec_channels[4], 7, 1)

        # Final Output Layer
        self.final = nn.Conv2d(dec_channels[4] + enc_channels[1], out_channels, kernel_size=3, padding=1, stride=1)

        # Initialize Spatial Attention Modules
        self.spatial_attn4 = SpatialAttention()
        self.spatial_attn3 = SpatialAttention()
        self.spatial_attn2 = SpatialAttention()
        self.spatial_attn1 = SpatialAttention()

    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout=0.2):
        """Convolutional Block with Dropout in Deeper Layers Only"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride), # In the next iteration i should introduce a stride here
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

    def upconv_block(self, in_channels, out_channels, kernel_size, stride, dropout=0.2):
        """Upsampling Block with Dropout in First Few Decoder Layers"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Dropout only for first few decoder layers
        if in_channels >= 256:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)


    def forward(self, x):
        """Forward pass with skip connections"""
        # Encoding
        e1 = self.enc1(x)  # (batch, 64, 1028, 175)
        e2 = self.enc2(e1) # (batch, 128, 514, 87)
        e3 = self.enc3(e2)  # (batch, 256, 257, 43)
        e4 = self.enc4(e3) # (batch, 512, 128, 21)

        # Bottleneck
        b = self.bottleneck(e4)  # (batch, 1024, 64, 10)

        # Decoding + Skip Connections with Spatial Attention
        d4 = self.dec4(b)  # (batch, 512, ?, ?)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=False)
        e4_attn = self.spatial_attn4(e4)  # Apply Spatial Attention
        d4 = torch.cat([d4, e4_attn], dim=1)

        d3 = self.dec3(d4)  # (batch, 256, ?, ?)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        e3_attn = self.spatial_attn3(e3)  # Apply Spatial Attention
        d3 = torch.cat([d3, e3_attn], dim=1)

        d2 = self.dec2(d3)  # (batch, 128, ?, ?)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        e2_attn = self.spatial_attn2(e2)  # Apply Spatial Attention
        d2 = torch.cat([d2, e2_attn], dim=1)

        d1 = self.dec1(d2)  # (batch, 64, ?, ?)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        e1_attn = self.spatial_attn1(e1)  # Apply Spatial Attention
        d1 = torch.cat([d1, e1_attn], dim=1)

        # Final Convolution (output denoised spectrogram)
        return F.interpolate(self.final(d1), size=(1025, 175), mode="bilinear", align_corners=False)
