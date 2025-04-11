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