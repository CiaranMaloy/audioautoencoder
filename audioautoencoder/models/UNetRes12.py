import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

## Masking UnetResidual
class ResBlock(nn.Module):
    def __init__(self, channels:int, num_groups:int, dropout_prob:float, kernel_size=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        # Adding proper padding calculation to maintain spatial dimensions
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x):
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.conv2(self.relu(self.gnorm2(r)))  # Fixed: use r instead of x for the second conv
        r = self.dropout(r)
        return r + x

class Attention(nn.Module):
    def __init__(self, channels: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(channels, channels*3)
        self.proj2 = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class ResLayer(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size=3,
                 attention=False,
                 num_groups=16,
                 dropout_prob=0.1,
                 num_heads=16,
                 upscale=False,
                 downscale=False):
        super().__init__()
        self.upscale = upscale
        self.downscale = downscale

        self.ResBlock1 = ResBlock(channels=channels, kernel_size=kernel_size, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(channels=channels, kernel_size=kernel_size, num_groups=num_groups, dropout_prob=dropout_prob)

        if upscale:
            self.conv = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0)
        elif downscale:
            self.conv = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)

        if attention:
            self.attention_layer = Attention(channels, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.ResBlock1(x)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x)

        if self.upscale:
            x = self.conv(x)
        elif self.downscale:
            x = self.conv(x)

        return x

class EnhancedSkipAttention(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, reduction_ratio=2):
        super().__init__()
        self.channels = encoder_channels

        # Channel attention for encoder features
        self.encoder_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(encoder_channels, encoder_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels // reduction_ratio, encoder_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Projection for decoder features to match encoder dimensions if needed
        self.decoder_proj = None
        if encoder_channels != decoder_channels:
            self.decoder_proj = nn.Conv2d(decoder_channels, encoder_channels, kernel_size=1)

        # Cross-attention between encoder and decoder features
        self.cross_attn = nn.Sequential(
            nn.Conv2d(encoder_channels*2, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, encoder_features, decoder_features):
        # Process decoder features if dimensions don't match
        if self.decoder_proj is not None:
            decoder_features = self.decoder_proj(decoder_features)

        # Apply channel attention to encoder features
        channel_attn = self.encoder_channel_attn(encoder_features)
        encoder_features = encoder_features * channel_attn

        # Concatenate encoder and decoder features
        combined = torch.cat([encoder_features, decoder_features], dim=1)

        # Generate attention weights for each feature set
        attn_weights = self.cross_attn(combined)
        encoder_weight, decoder_weight = torch.split(attn_weights, 1, dim=1)

        # Apply weights and combine features
        result = encoder_features * encoder_weight + decoder_features * decoder_weight

        return result

class UNetRes12(nn.Module):
    # Update from UnetConv6, moving to a masking model, which hopefully works better
    def __init__(self, in_channels=9, out_channels=4, num_groups=16):
        super().__init__()

        channels = 32

        # Fixed input layer with proper padding calculation for given kernel size
        self.sigmoid = nn.Sigmoid()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder (Downsampling) - using standard kernel sizes with proper padding
        self.enc1 = ResLayer(channels, kernel_size=3, downscale=True)
        self.enc2 = ResLayer(channels * 2, kernel_size=3, downscale=True)
        self.enc3 = ResLayer(channels * 4, kernel_size=3, downscale=True, attention=True)
        self.enc4 = ResLayer(channels * 8, kernel_size=3, downscale=True, dropout_prob=0.2)

        # Bottleneck
        self.bottleneck_in = ResLayer(channels * 16, kernel_size=3, dropout_prob=0.3)
        self.resattention = ResLayer(channels * 16, kernel_size=3, attention=True, dropout_prob=0.4)
        self.bottleneck_out = ResLayer(channels * 16, kernel_size=3, dropout_prob=0.3)

        # Decoder (Upsampling) - using standard kernel sizes
        self.dec4 = ResLayer(channels * 16, kernel_size=3, upscale=True, dropout_prob=0.2)
        self.dec3 = ResLayer(channels * 8, kernel_size=3, upscale=True, attention=True)
        self.dec2 = ResLayer(channels * 4, kernel_size=3, upscale=True)
        self.dec1 = ResLayer(channels * 2, kernel_size=3, upscale=True, attention=True)

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

        # Initialize Spatial Attention Modules
        self.attn4 = EnhancedSkipAttention(channels * 16, channels * 16)
        self.attn3 = EnhancedSkipAttention(channels * 8, channels * 8)
        self.attn2 = EnhancedSkipAttention(channels * 4, channels * 4)
        self.attn1 = EnhancedSkipAttention(channels * 2, channels * 2)

    def forward(self, x, return_mask_only=False):
        """Forward pass with skip connections"""
        input_shape = x.shape[2:]  # Remember original input spatial dimensions

        # Encoding
        input_features = self.input_layer(x)
        e1 = self.enc1(input_features)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck_in(e4)
        b = self.resattention(b)
        b = self.bottleneck_out(b)

        # Decoding with proper feature alignment
        # For decoder stage 4
        b = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.attn4(e4, b) # i should try and replace these with concatenations in the future
        d4 = self.dec4(d4)

        # For decoder stage 3
        d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.attn3(e3, d4)
        d3 = self.dec3(d3)

        # For decoder stage 2
        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.attn2(e2, d3)
        d2 = self.dec2(d2)

        # For decoder stage 1
        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.attn1(e1, d2)
        d1 = self.dec1(d1)

        # Final output with bilinear interpolation to match input size
        mask = self.output_layer(d1)
        mask = F.interpolate(mask, size=input_shape, mode="bilinear", align_corners=False)

        if return_mask_only:
            return self.sigmoid(mask)
        else:
            return x * self.sigmoid(mask)
        

'''
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
UNetRes12                                [2, 4, 256, 175]          --
├─Sequential: 1-1                        [2, 32, 128, 88]          --
│    └─Conv2d: 2-1                       [2, 32, 128, 88]          1,184
│    └─GroupNorm: 2-2                    [2, 32, 128, 88]          64
│    └─LeakyReLU: 2-3                    [2, 32, 128, 88]          --
│    └─Conv2d: 2-4                       [2, 32, 128, 88]          9,248
│    └─GroupNorm: 2-5                    [2, 32, 128, 88]          64
│    └─LeakyReLU: 2-6                    [2, 32, 128, 88]          --
├─ResLayer: 1-2                          [2, 64, 64, 44]           --
│    └─ResBlock: 2-7                     [2, 32, 128, 88]          --
│    │    └─GroupNorm: 3-1               [2, 32, 128, 88]          64
│    │    └─ReLU: 3-2                    [2, 32, 128, 88]          --
│    │    └─Conv2d: 3-3                  [2, 32, 128, 88]          9,248
│    │    └─GroupNorm: 3-4               [2, 32, 128, 88]          64
│    │    └─ReLU: 3-5                    [2, 32, 128, 88]          --
│    │    └─Conv2d: 3-6                  [2, 32, 128, 88]          9,248
│    │    └─Dropout: 3-7                 [2, 32, 128, 88]          --
│    └─ResBlock: 2-8                     [2, 32, 128, 88]          --
│    │    └─GroupNorm: 3-8               [2, 32, 128, 88]          64
│    │    └─ReLU: 3-9                    [2, 32, 128, 88]          --
│    │    └─Conv2d: 3-10                 [2, 32, 128, 88]          9,248
│    │    └─GroupNorm: 3-11              [2, 32, 128, 88]          64
│    │    └─ReLU: 3-12                   [2, 32, 128, 88]          --
│    │    └─Conv2d: 3-13                 [2, 32, 128, 88]          9,248
│    │    └─Dropout: 3-14                [2, 32, 128, 88]          --
│    └─Conv2d: 2-9                       [2, 64, 64, 44]           18,496
├─ResLayer: 1-3                          [2, 128, 32, 22]          --
│    └─ResBlock: 2-10                    [2, 64, 64, 44]           --
│    │    └─GroupNorm: 3-15              [2, 64, 64, 44]           128
│    │    └─ReLU: 3-16                   [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-17                 [2, 64, 64, 44]           36,928
│    │    └─GroupNorm: 3-18              [2, 64, 64, 44]           128
│    │    └─ReLU: 3-19                   [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-20                 [2, 64, 64, 44]           36,928
│    │    └─Dropout: 3-21                [2, 64, 64, 44]           --
│    └─ResBlock: 2-11                    [2, 64, 64, 44]           --
│    │    └─GroupNorm: 3-22              [2, 64, 64, 44]           128
│    │    └─ReLU: 3-23                   [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-24                 [2, 64, 64, 44]           36,928
│    │    └─GroupNorm: 3-25              [2, 64, 64, 44]           128
│    │    └─ReLU: 3-26                   [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-27                 [2, 64, 64, 44]           36,928
│    │    └─Dropout: 3-28                [2, 64, 64, 44]           --
│    └─Conv2d: 2-12                      [2, 128, 32, 22]          73,856
├─ResLayer: 1-4                          [2, 256, 16, 11]          --
│    └─ResBlock: 2-13                    [2, 128, 32, 22]          --
│    │    └─GroupNorm: 3-29              [2, 128, 32, 22]          256
│    │    └─ReLU: 3-30                   [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-31                 [2, 128, 32, 22]          147,584
│    │    └─GroupNorm: 3-32              [2, 128, 32, 22]          256
│    │    └─ReLU: 3-33                   [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-34                 [2, 128, 32, 22]          147,584
│    │    └─Dropout: 3-35                [2, 128, 32, 22]          --
│    └─Attention: 2-14                   [2, 128, 32, 22]          --
│    │    └─Linear: 3-36                 [2, 704, 384]             49,536
│    │    └─Linear: 3-37                 [2, 32, 22, 128]          16,512
│    └─ResBlock: 2-15                    [2, 128, 32, 22]          --
│    │    └─GroupNorm: 3-38              [2, 128, 32, 22]          256
│    │    └─ReLU: 3-39                   [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-40                 [2, 128, 32, 22]          147,584
│    │    └─GroupNorm: 3-41              [2, 128, 32, 22]          256
│    │    └─ReLU: 3-42                   [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-43                 [2, 128, 32, 22]          147,584
│    │    └─Dropout: 3-44                [2, 128, 32, 22]          --
│    └─Conv2d: 2-16                      [2, 256, 16, 11]          295,168
├─ResLayer: 1-5                          [2, 512, 8, 6]            --
│    └─ResBlock: 2-17                    [2, 256, 16, 11]          --
│    │    └─GroupNorm: 3-45              [2, 256, 16, 11]          512
│    │    └─ReLU: 3-46                   [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-47                 [2, 256, 16, 11]          590,080
│    │    └─GroupNorm: 3-48              [2, 256, 16, 11]          512
│    │    └─ReLU: 3-49                   [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-50                 [2, 256, 16, 11]          590,080
│    │    └─Dropout: 3-51                [2, 256, 16, 11]          --
│    └─ResBlock: 2-18                    [2, 256, 16, 11]          --
│    │    └─GroupNorm: 3-52              [2, 256, 16, 11]          512
│    │    └─ReLU: 3-53                   [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-54                 [2, 256, 16, 11]          590,080
│    │    └─GroupNorm: 3-55              [2, 256, 16, 11]          512
│    │    └─ReLU: 3-56                   [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-57                 [2, 256, 16, 11]          590,080
│    │    └─Dropout: 3-58                [2, 256, 16, 11]          --
│    └─Conv2d: 2-19                      [2, 512, 8, 6]            1,180,160
├─ResLayer: 1-6                          [2, 512, 8, 6]            --
│    └─ResBlock: 2-20                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-59              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-60                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-61                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-62              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-63                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-64                 [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-65                [2, 512, 8, 6]            --
│    └─ResBlock: 2-21                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-66              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-67                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-68                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-69              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-70                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-71                 [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-72                [2, 512, 8, 6]            --
├─ResLayer: 1-7                          [2, 512, 8, 6]            --
│    └─ResBlock: 2-22                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-73              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-74                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-75                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-76              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-77                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-78                 [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-79                [2, 512, 8, 6]            --
│    └─Attention: 2-23                   [2, 512, 8, 6]            --
│    │    └─Linear: 3-80                 [2, 48, 1536]             787,968
│    │    └─Linear: 3-81                 [2, 8, 6, 512]            262,656
│    └─ResBlock: 2-24                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-82              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-83                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-84                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-85              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-86                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-87                 [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-88                [2, 512, 8, 6]            --
├─ResLayer: 1-8                          [2, 512, 8, 6]            --
│    └─ResBlock: 2-25                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-89              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-90                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-91                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-92              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-93                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-94                 [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-95                [2, 512, 8, 6]            --
│    └─ResBlock: 2-26                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-96              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-97                   [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-98                 [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-99              [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-100                  [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-101                [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-102               [2, 512, 8, 6]            --
├─EnhancedSkipAttention: 1-9             [2, 512, 8, 6]            --
│    └─Sequential: 2-27                  [2, 512, 1, 1]            --
│    │    └─AdaptiveAvgPool2d: 3-103     [2, 512, 1, 1]            --
│    │    └─Conv2d: 3-104                [2, 256, 1, 1]            131,328
│    │    └─ReLU: 3-105                  [2, 256, 1, 1]            --
│    │    └─Conv2d: 3-106                [2, 512, 1, 1]            131,584
│    │    └─Sigmoid: 3-107               [2, 512, 1, 1]            --
│    └─Sequential: 2-28                  [2, 2, 8, 6]              --
│    │    └─Conv2d: 3-108                [2, 2, 8, 6]              18,434
│    │    └─Sigmoid: 3-109               [2, 2, 8, 6]              --
├─ResLayer: 1-10                         [2, 256, 16, 12]          --
│    └─ResBlock: 2-29                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-110             [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-111                  [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-112                [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-113             [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-114                  [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-115                [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-116               [2, 512, 8, 6]            --
│    └─ResBlock: 2-30                    [2, 512, 8, 6]            --
│    │    └─GroupNorm: 3-117             [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-118                  [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-119                [2, 512, 8, 6]            2,359,808
│    │    └─GroupNorm: 3-120             [2, 512, 8, 6]            1,024
│    │    └─ReLU: 3-121                  [2, 512, 8, 6]            --
│    │    └─Conv2d: 3-122                [2, 512, 8, 6]            2,359,808
│    │    └─Dropout: 3-123               [2, 512, 8, 6]            --
│    └─ConvTranspose2d: 2-31             [2, 256, 16, 12]          524,544
├─EnhancedSkipAttention: 1-11            [2, 256, 16, 11]          --
│    └─Sequential: 2-32                  [2, 256, 1, 1]            --
│    │    └─AdaptiveAvgPool2d: 3-124     [2, 256, 1, 1]            --
│    │    └─Conv2d: 3-125                [2, 128, 1, 1]            32,896
│    │    └─ReLU: 3-126                  [2, 128, 1, 1]            --
│    │    └─Conv2d: 3-127                [2, 256, 1, 1]            33,024
│    │    └─Sigmoid: 3-128               [2, 256, 1, 1]            --
│    └─Sequential: 2-33                  [2, 2, 16, 11]            --
│    │    └─Conv2d: 3-129                [2, 2, 16, 11]            9,218
│    │    └─Sigmoid: 3-130               [2, 2, 16, 11]            --
├─ResLayer: 1-12                         [2, 128, 32, 22]          --
│    └─ResBlock: 2-34                    [2, 256, 16, 11]          --
│    │    └─GroupNorm: 3-131             [2, 256, 16, 11]          512
│    │    └─ReLU: 3-132                  [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-133                [2, 256, 16, 11]          590,080
│    │    └─GroupNorm: 3-134             [2, 256, 16, 11]          512
│    │    └─ReLU: 3-135                  [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-136                [2, 256, 16, 11]          590,080
│    │    └─Dropout: 3-137               [2, 256, 16, 11]          --
│    └─Attention: 2-35                   [2, 256, 16, 11]          --
│    │    └─Linear: 3-138                [2, 176, 768]             197,376
│    │    └─Linear: 3-139                [2, 16, 11, 256]          65,792
│    └─ResBlock: 2-36                    [2, 256, 16, 11]          --
│    │    └─GroupNorm: 3-140             [2, 256, 16, 11]          512
│    │    └─ReLU: 3-141                  [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-142                [2, 256, 16, 11]          590,080
│    │    └─GroupNorm: 3-143             [2, 256, 16, 11]          512
│    │    └─ReLU: 3-144                  [2, 256, 16, 11]          --
│    │    └─Conv2d: 3-145                [2, 256, 16, 11]          590,080
│    │    └─Dropout: 3-146               [2, 256, 16, 11]          --
│    └─ConvTranspose2d: 2-37             [2, 128, 32, 22]          131,200
├─EnhancedSkipAttention: 1-13            [2, 128, 32, 22]          --
│    └─Sequential: 2-38                  [2, 128, 1, 1]            --
│    │    └─AdaptiveAvgPool2d: 3-147     [2, 128, 1, 1]            --
│    │    └─Conv2d: 3-148                [2, 64, 1, 1]             8,256
│    │    └─ReLU: 3-149                  [2, 64, 1, 1]             --
│    │    └─Conv2d: 3-150                [2, 128, 1, 1]            8,320
│    │    └─Sigmoid: 3-151               [2, 128, 1, 1]            --
│    └─Sequential: 2-39                  [2, 2, 32, 22]            --
│    │    └─Conv2d: 3-152                [2, 2, 32, 22]            4,610
│    │    └─Sigmoid: 3-153               [2, 2, 32, 22]            --
├─ResLayer: 1-14                         [2, 64, 64, 44]           --
│    └─ResBlock: 2-40                    [2, 128, 32, 22]          --
│    │    └─GroupNorm: 3-154             [2, 128, 32, 22]          256
│    │    └─ReLU: 3-155                  [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-156                [2, 128, 32, 22]          147,584
│    │    └─GroupNorm: 3-157             [2, 128, 32, 22]          256
│    │    └─ReLU: 3-158                  [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-159                [2, 128, 32, 22]          147,584
│    │    └─Dropout: 3-160               [2, 128, 32, 22]          --
│    └─ResBlock: 2-41                    [2, 128, 32, 22]          --
│    │    └─GroupNorm: 3-161             [2, 128, 32, 22]          256
│    │    └─ReLU: 3-162                  [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-163                [2, 128, 32, 22]          147,584
│    │    └─GroupNorm: 3-164             [2, 128, 32, 22]          256
│    │    └─ReLU: 3-165                  [2, 128, 32, 22]          --
│    │    └─Conv2d: 3-166                [2, 128, 32, 22]          147,584
│    │    └─Dropout: 3-167               [2, 128, 32, 22]          --
│    └─ConvTranspose2d: 2-42             [2, 64, 64, 44]           32,832
├─EnhancedSkipAttention: 1-15            [2, 64, 64, 44]           --
│    └─Sequential: 2-43                  [2, 64, 1, 1]             --
│    │    └─AdaptiveAvgPool2d: 3-168     [2, 64, 1, 1]             --
│    │    └─Conv2d: 3-169                [2, 32, 1, 1]             2,080
│    │    └─ReLU: 3-170                  [2, 32, 1, 1]             --
│    │    └─Conv2d: 3-171                [2, 64, 1, 1]             2,112
│    │    └─Sigmoid: 3-172               [2, 64, 1, 1]             --
│    └─Sequential: 2-44                  [2, 2, 64, 44]            --
│    │    └─Conv2d: 3-173                [2, 2, 64, 44]            2,306
│    │    └─Sigmoid: 3-174               [2, 2, 64, 44]            --
├─ResLayer: 1-16                         [2, 32, 128, 88]          --
│    └─ResBlock: 2-45                    [2, 64, 64, 44]           --
│    │    └─GroupNorm: 3-175             [2, 64, 64, 44]           128
│    │    └─ReLU: 3-176                  [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-177                [2, 64, 64, 44]           36,928
│    │    └─GroupNorm: 3-178             [2, 64, 64, 44]           128
│    │    └─ReLU: 3-179                  [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-180                [2, 64, 64, 44]           36,928
│    │    └─Dropout: 3-181               [2, 64, 64, 44]           --
│    └─Attention: 2-46                   [2, 64, 64, 44]           --
│    │    └─Linear: 3-182                [2, 2816, 192]            12,480
│    │    └─Linear: 3-183                [2, 64, 44, 64]           4,160
│    └─ResBlock: 2-47                    [2, 64, 64, 44]           --
│    │    └─GroupNorm: 3-184             [2, 64, 64, 44]           128
│    │    └─ReLU: 3-185                  [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-186                [2, 64, 64, 44]           36,928
│    │    └─GroupNorm: 3-187             [2, 64, 64, 44]           128
│    │    └─ReLU: 3-188                  [2, 64, 64, 44]           --
│    │    └─Conv2d: 3-189                [2, 64, 64, 44]           36,928
│    │    └─Dropout: 3-190               [2, 64, 64, 44]           --
│    └─ConvTranspose2d: 2-48             [2, 32, 128, 88]          8,224
├─Sequential: 1-17                       [2, 4, 255, 175]          --
│    └─ConvTranspose2d: 2-49             [2, 32, 255, 175]         9,248
│    └─LeakyReLU: 2-50                   [2, 32, 255, 175]         --
│    └─Conv2d: 2-51                      [2, 4, 255, 175]          1,156
├─Sigmoid: 1-18                          [2, 4, 256, 175]          --
==========================================================================================
Total params: 48,080,556
Trainable params: 48,080,556
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 11.82
==========================================================================================
Input size (MB): 1.43
Forward/backward pass size (MB): 226.45
Params size (MB): 192.32
Estimated Total Size (MB): 420.21
==========================================================================================
'''