import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
import math

## Masking Unet
# enhanced attention on skip connections
# deep 3 headded self attention in bottleneck
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]

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

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))  # Fixed: use r instead of x for the second conv
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

class LongformerAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout_prob: float, window_size: int = 1025, global_tokens: int = 512):
        """
        Longformer attention that uses a sliding window + global attention mechanism.
        
        Args:
            channels: Input channel dimension
            num_heads: Number of attention heads
            dropout_prob: Dropout probability
            window_size: Size of the local attention window
            global_tokens: Number of global tokens to use (typically corners or other important positions)
        """
        super().__init__()
        self.proj1 = nn.Linear(channels, channels * 3)
        self.proj2 = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.head_dim = channels // num_heads
        
    def forward(self, x):
        h, w = x.shape[2:]
        seq_len = h * w
        batch_size = x.shape[0]
        
        # Reshape to sequence format
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Project to query, key, value
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        
        # Create attention mask for sliding window
        # Start with a mask that allows local attention within the window
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        
        # Fill in the local attention windows
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        
        # Add global attention tokens (e.g., corners of the image)
        # For simplicity, we'll use the first `global_tokens` positions
        if self.global_tokens > 0:
            global_indices = [0, w-1, seq_len-w, seq_len-1][:self.global_tokens]  # Corners
            
            # Global tokens attend to all positions
            for idx in global_indices:
                mask[idx, :] = 1
                mask[:, idx] = 1
        
        # Convert mask to attention mask format (0 for attended positions, -inf for masked positions)
        attn_mask = torch.zeros_like(mask)
        attn_mask[mask == 0] = float('-inf')
        
        # Compute attention with the mask
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Apply the attention mask
        attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_prob, training=self.training)
        
        # Apply attention to values
        x = torch.matmul(attn, v)
        
        # Reshape back to original format
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class ResLayer(nn.Module):
    def __init__(self, 
                 channels, 
                 kernel_size=3, 
                 attention=False, 
                 long_attention=False, 
                 lin_attention=False,
                 num_groups=16, 
                 dropout_prob=0.1, 
                 num_heads=8, 
                 upscale=False, 
                 downscale=False):
        super().__init__()
        self.upscale = upscale
        self.downscale = downscale

        # Calculate proper padding
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        
        self.ResBlock1 = ResBlock(channels=channels, kernel_size=kernel_size, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(channels=channels, kernel_size=kernel_size, num_groups=num_groups, dropout_prob=dropout_prob)

        if upscale:
            self.conv = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0)
        elif downscale:
            self.conv = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)

        if attention:
            self.attention_layer = Attention(channels, num_heads=num_heads, dropout_prob=dropout_prob)
        if long_attention:
            self.attention_layer = LongformerAttention(channels, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)

        if self.upscale:
            x = self.conv(x)
        elif self.downscale:
            x = self.conv(x)

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

class UNetConv11(nn.Module):
    # Update from UnetConv6, moving to a masking model, which hopefully works better
    def __init__(self, in_channels=4, out_channels=4, time_steps=1000):
        super().__init__()

        channels = 64

        # Fixed input layer with proper padding calculation for given kernel size
        self.sigmoid = nn.Sigmoid()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=(20, 10), padding=1, stride=2),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(10, 5), padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder (Downsampling) - using standard kernel sizes with proper padding
        self.enc1 = ResLayer(channels, kernel_size=7, downscale=True)
        self.enc2 = ResLayer(channels * 2, kernel_size=5, downscale=True)
        self.enc3 = ResLayer(channels * 4, kernel_size=3, downscale=True, attention=True)
        self.enc4 = ResLayer(channels * 8, kernel_size=3, downscale=True, dropout_prob=0.2)

        # Bottleneck
        self.bottleneck_in = ResLayer(channels * 16, kernel_size=3, dropout_prob=0.3)
        self.resattention = ResLayer(channels * 16, kernel_size=3, attention=True, dropout_prob=0.4)
        self.bottleneck_out = ResLayer(channels * 16, kernel_size=3, dropout_prob=0.3)

        # Decoder (Upsampling) - using standard kernel sizes
        self.dec4 = ResLayer(channels * 16, kernel_size=3, upscale=True, dropout_prob=0.2)
        self.dec3 = ResLayer(channels * 8, kernel_size=3, upscale=True, attention=True)
        self.dec2 = ResLayer(channels * 4, kernel_size=5, upscale=True)
        self.dec1 = ResLayer(channels * 2, kernel_size=7, upscale=True)

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(10, 5), padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, out_channels, kernel_size=(20, 10), padding=1, stride=1)
        )

        # Initialize Spatial Attention Modules
        self.attn4 = EnhancedSkipAttention(channels * 16, channels * 16)
        self.attn3 = EnhancedSkipAttention(channels * 8, channels * 8)
        self.attn2 = EnhancedSkipAttention(channels * 4, channels * 4)
        self.attn1 = EnhancedSkipAttention(channels * 2, channels * 2)

        # embeddings 
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=channels * 16)

    def forward(self, x, t=0):
        """Forward pass with skip connections"""
        input_shape = x.shape[2:]  # Remember original input spatial dimensions
        embeddings = self.embeddings(x, t)

        # Encoding
        input_features = self.input_layer(x)
        e1 = self.enc1(input_features, embeddings)
        e2 = self.enc2(e1, embeddings)
        e3 = self.enc3(e2, embeddings)
        e4 = self.enc4(e3, embeddings)

        # Bottleneck
        b = self.bottleneck_in(e4, embeddings)
        b = self.resattention(b, embeddings)
        b = self.bottleneck_out(b, embeddings)

        # Decoding with proper feature alignment
        # For decoder stage 4
        b = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.attn4(e4, b)
        d4 = self.dec4(d4, embeddings)

        # For decoder stage 3
        d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.attn3(e3, d4)
        d3 = self.dec3(d3, embeddings)

        # For decoder stage 2
        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.attn2(e2, d3)
        d2 = self.dec2(d2, embeddings)

        # For decoder stage 1
        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.attn1(e1, d2)
        d1 = self.dec1(d1, embeddings)

        # Final output with bilinear interpolation to match input size
        mask = self.output_layer(d1)
        mask = F.interpolate(mask, size=input_shape, mode="bilinear", align_corners=False)
        
        # Apply mask to the first 4 channels of input
        return mask