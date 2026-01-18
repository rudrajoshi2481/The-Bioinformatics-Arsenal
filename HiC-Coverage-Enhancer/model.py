"""
Hi-C Enhancement Deep Learning Model
====================================

Architecture: Hybrid UNet-Transformer with Learnable Wavelet Decomposition

Components:
1. Learnable Wavelet Layer: Decomposes input into low/high frequency
2. 2D Positional Encoding: Adds spatial awareness
3. Dual-path UNet Encoder: Separate encoders for structure & details
4. Transformer: Captures global context with multi-head attention
5. UNet Decoder: Reconstructs enhanced output

Author: Bioinformatics Deep Learning Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for Hi-C matrices"""
    
    def __init__(self, channels, height=20, width=20, max_freq=10000):
        super().__init__()
        
        pe = torch.zeros(channels, height, width)
        
        # Row encodings
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width).float()
        # Col encodings
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1).float()
        
        # Frequencies
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                            (-math.log(max_freq) / channels))
        
        # Apply sin/cos to positions
        pe[0::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[1::2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe.unsqueeze(0)


class LearnableWaveletLayer(nn.Module):
    """Learnable wavelet-like decomposition"""
    
    def __init__(self):
        super().__init__()
        
        # Learnable filters
        self.low_filter = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.high_filter = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1, bias=False)
        
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        """Initialize with Haar-like wavelets"""
        # Low-pass (averaging)
        self.low_filter.weight.data = torch.ones(1, 1, 3, 3) / 9
        
        # High-pass filters
        h_filter = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        v_filter = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        d_filter = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=torch.float32)
        
        self.high_filter.weight.data[0, 0] = h_filter
        self.high_filter.weight.data[1, 0] = v_filter
        self.high_filter.weight.data[2, 0] = d_filter
    
    def forward(self, x):
        low = self.low_filter(x)   # (B, 1, 20, 20)
        high = self.high_filter(x)  # (B, 3, 20, 20)
        return low, high


class HiCEnhancementModel(nn.Module):
    """
    Complete Hi-C Enhancement Model
    
    Architecture:
    1. Learnable Wavelet Decomposition
    2. Dual-path UNet Encoder (low-freq + high-freq)
    3. 2D Positional Encoding
    4. Transformer for global context
    5. UNet Decoder with skip connections
    """
    
    def __init__(self, 
                 base_channels=64,
                 num_transformer_layers=4,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        # Wavelet decomposition
        self.wavelet_decomp = LearnableWaveletLayer()
        
        # Encoder for low-frequency (structure)
        self.low_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder for high-frequency (details)
        self.high_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Combine features
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2D Positional Encoding
        self.pos_encoding = PositionalEncoding2D(
            channels=base_channels,
            height=20,
            width=20
        )
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=base_channels,
                nhead=num_heads,
                dim_feedforward=base_channels * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels//4, 3, padding=1),
            nn.BatchNorm2d(base_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//4, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 40, 40)
        Returns:
            enhanced: (batch, 1, 40, 40)
        """
        # 1. Wavelet decomposition
        low, high = self.wavelet_decomp(x)  # (B,1,20,20), (B,3,20,20)
        
        # 2. Encode frequency bands
        low_feat = self.low_encoder(low)    # (B, 64, 20, 20)
        high_feat = self.high_encoder(high)  # (B, 64, 20, 20)
        
        # 3. Fuse features
        combined = torch.cat([low_feat, high_feat], dim=1)  # (B, 128, 20, 20)
        fused = self.fusion(combined)  # (B, 64, 20, 20)
        
        # 4. Add positional encoding
        fused_with_pos = self.pos_encoding(fused)  # (B, 64, 20, 20)
        
        # 5. Flatten for transformer
        b, c, h, w = fused_with_pos.shape
        flat = fused_with_pos.view(b, c, -1).transpose(1, 2)  # (B, 400, 64)
        
        # 6. Global attention
        attended = self.transformer(flat)  # (B, 400, 64)
        
        # 7. Reshape
        spatial = attended.transpose(1, 2).view(b, c, h, w)  # (B, 64, 20, 20)
        
        # 8. Decode
        enhanced = self.decoder(spatial)  # (B, 1, 40, 40)
        
        return enhanced


# Test model
if __name__ == "__main__":
    model = HiCEnhancementModel()
    x = torch.randn(2, 1, 40, 40)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
