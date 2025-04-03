import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 

## Masking Unet
# enhanced attention on skip connections
# deep 3 headded self attention in bottleneck
class ResBlock(nn.Module):
    def __init__(self, channels:int, num_groups:int, dropout_prob:float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x):
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(x)))
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
    

class ResAttentionLayer(nn.Module):
    def __init__(self, channels, attention=True, num_groups=32, dropout_prob=0.1, num_heads=8):
        super().__init__()
        self.ResBlock1 = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob)

        if attention:
            self.attention_layer = Attention(channels, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.ResBlock1(x)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x)
        return x

class EnhancedSkipAttention(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, reduction_ratio=4):
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

class UNetConv10(nn.Module):
    # Update from UnetConv6, moving to a masking model, which hopefully works better
    def __init__(self, in_channels=9, out_channels=4):
        super().__init__()

        a = 2
        A, B, C, D = 64, 128, 256, 512
        bottleneck_channels = 1024

        # function
        self.sigmoid = nn.Sigmoid()

        # Encoder (Downsampling)
        enc_channels = [A, B, C, D]
        dec_channels = [D, C, B, A]

        self.enc1 = self.conv_block(in_channels, enc_channels[0], (10, 3), 2)
        self.enc2 = self.conv_block(enc_channels[0], enc_channels[1], 5, 2)
        self.enc3 = self.conv_block(enc_channels[1], enc_channels[2], 3, 2)
        self.enc4 = self.conv_block(enc_channels[2], enc_channels[3], 3, 2)

        # Bottleneck
        self.bottleneck_in = self.conv_block(enc_channels[3], bottleneck_channels, 3, 2)
        self.resattention = ResAttentionLayer(bottleneck_channels, attention=True)
        self.bottleneck_out = self.upconv_block(bottleneck_channels, dec_channels[0], 3, 2)

        # Decoder (Upsampling)
        self.dec4 = self.upconv_block(dec_channels[0], dec_channels[1], 3, 2)
        self.dec3 = self.upconv_block(dec_channels[1], dec_channels[2], 3, 2)
        self.dec2 = self.upconv_block(dec_channels[2], dec_channels[3], 5, 2)
        self.dec1 = self.upconv_block(dec_channels[3], out_channels, (10, 3), 2)

        # Initialize Spatial Attention Modules
        self.attn4 = EnhancedSkipAttention(enc_channels[3], dec_channels[0])
        self.attn3 = EnhancedSkipAttention(enc_channels[2], dec_channels[1])
        self.attn2 = EnhancedSkipAttention(enc_channels[1], dec_channels[2])
        self.attn1 = EnhancedSkipAttention(enc_channels[0], dec_channels[3])

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
        b = self.bottleneck_in(e4)  # (batch, 1024, 64, 10)
        b = self.resattention(b)  # (batch, 1024, 64, 10)
        b = self.bottleneck_out(b)  # (batch, 512, 64, 10)

        # Decoding + Skip Connections with Spatial Attention
        b = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4_attn = self.attn4(e4, b)  # Apply Spatial Attention
        d4 = self.dec4(d4_attn)  # (batch, 512, ?, ?)

        d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3_attn = self.attn3(e3, d4)  # Apply Spatial Attention
        d3 = self.dec3(d3_attn)  # (batch, 256, ?, ?)

        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2_attn = self.attn2(e2, d3)  # Apply Spatial Attention
        d2 = self.dec2(d2_attn)  # (batch, 128, ?, ?)

        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1_attn = self.attn1(e1, d2)  # Apply Spatial Attention
        d1 = self.dec1(d1_attn)  # (batch, 64, ?, ?)

        # Final Convolution (output denoised spectrogram)
        mask = F.interpolate(d1, size=(1025 // 4, 175), mode="bilinear", align_corners=False)
        return x[:, :4] * self.sigmoid(mask)
