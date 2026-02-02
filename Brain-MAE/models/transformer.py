"""
Transformer Architecture for Brain MAE
Modular transformer blocks with 3D positional encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config


class SinusoidalPositionalEncoding3D(nn.Module):
    """
    3D Sinusoidal Positional Encoding for brain patches.
    Encodes (x, y, z) grid positions into continuous embeddings.
    """
    
    def __init__(self, embed_dim: int, grid_size: Tuple[int, int, int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Pre-compute positional encodings
        n_patches = grid_size[0] * grid_size[1] * grid_size[2]
        pe = self._create_3d_sinusoidal_encoding(grid_size, embed_dim)
        self.register_buffer("pe", pe)  # (n_patches, embed_dim)
    
    def _create_3d_sinusoidal_encoding(
        self, 
        grid_size: Tuple[int, int, int], 
        embed_dim: int
    ) -> torch.Tensor:
        """Create 3D sinusoidal positional encoding"""
        gx, gy, gz = grid_size
        
        # Divide embedding dimension among 3 axes
        dim_per_axis = embed_dim // 3
        remainder = embed_dim % 3
        
        dims = [dim_per_axis + (1 if i < remainder else 0) for i in range(3)]
        
        encodings = []
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    pos_enc = []
                    for pos, dim in zip([i, j, k], dims):
                        enc = self._sinusoidal_encoding(pos, dim)
                        pos_enc.append(enc)
                    encodings.append(torch.cat(pos_enc))
        
        return torch.stack(encodings)  # (n_patches, embed_dim)
    
    def _sinusoidal_encoding(self, position: int, dim: int) -> torch.Tensor:
        """Standard sinusoidal encoding for a single position"""
        encoding = torch.zeros(dim)
        for i in range(0, dim, 2):
            div_term = math.exp(i * (-math.log(10000.0) / dim))
            encoding[i] = math.sin(position * div_term)
            if i + 1 < dim:
                encoding[i + 1] = math.cos(position * div_term)
        return encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, n_patches, embed_dim)
        
        Returns:
            x + positional encoding
        """
        return x + self.pe.unsqueeze(0)


class LearnablePositionalEncoding3D(nn.Module):
    """Learnable 3D positional encoding (alternative to sinusoidal)"""
    
    def __init__(self, n_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with pre-norm (more stable training).
    
    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
             |__________________________|    |___________________|
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x), mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks for encoding"""
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Lightweight Transformer decoder for MAE reconstruction"""
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test transformer components
    print("Testing Transformer Components")
    print("=" * 60)
    
    batch_size = 4
    n_patches = 192
    embed_dim = 256
    
    # Test positional encoding
    print("\n1. Positional Encoding")
    pos_enc = SinusoidalPositionalEncoding3D(embed_dim, (8, 8, 3))
    x = torch.randn(batch_size, n_patches, embed_dim)
    x_pos = pos_enc(x)
    print(f"   Input: {x.shape} -> Output: {x_pos.shape}")
    print(f"   Positional encoding added: {(x_pos - x).abs().mean():.4f}")
    
    # Test attention
    print("\n2. Multi-Head Attention")
    attn = MultiHeadAttention(embed_dim, num_heads=4)
    out = attn(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {count_parameters(attn):,}")
    
    # Test transformer block
    print("\n3. Transformer Block")
    block = TransformerBlock(embed_dim, num_heads=4, mlp_ratio=2.0)
    out = block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {count_parameters(block):,}")
    
    # Test encoder
    print("\n4. Transformer Encoder (4 layers)")
    encoder = TransformerEncoder(embed_dim, depth=4, num_heads=4, mlp_ratio=2.0)
    out = encoder(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {count_parameters(encoder):,}")
    
    # Test decoder
    print("\n5. Transformer Decoder (2 layers)")
    decoder = TransformerDecoder(128, depth=2, num_heads=4, mlp_ratio=2.0)
    x_dec = torch.randn(batch_size, n_patches, 128)
    out = decoder(x_dec)
    print(f"   Input: {x_dec.shape} -> Output: {out.shape}")
    print(f"   Parameters: {count_parameters(decoder):,}")
    
    print("\n" + "=" * 60)
    print("All transformer components working!")
