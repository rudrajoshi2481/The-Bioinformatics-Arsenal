"""
Transformer modules for Point-MAE.

Contains:
- TransformerEncoder: Stack of transformer blocks for encoding
- TransformerDecoder: Stack of transformer blocks for decoding
- MaskTransformer: Masked transformer encoder for MAE
"""

import torch
import torch.nn as nn
import numpy as np
import random

from .attention import Block
from .encoder import Encoder
from .utils import trunc_normal_


class TransformerEncoder(nn.Module):
    """Transformer encoder - stack of transformer blocks."""
    
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, 
                  drop_path=dpr[i])
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder - stack of transformer blocks with final norm."""
    
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, 
                  drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)
        x = self.norm(x[:, -return_token_num:])
        return x


class MaskTransformer(nn.Module):
    """Masked Transformer Encoder for Point-MAE."""
    
    def __init__(self, trans_dim=384, depth=12, num_heads=6, encoder_dims=384,
                 input_channel=6, mask_ratio=0.6, mask_type='rand', drop_path_rate=0.1):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.trans_dim = trans_dim
        
        # Point encoder
        self.encoder = Encoder(encoder_channel=encoder_dims, input_channel=input_channel)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        
        # Transformer encoder
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )
        
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_rand(self, center, noaug=False):
        """Random masking."""
        B, G, _ = center.shape
        
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)

        num_mask = int(self.mask_ratio * G)
        
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool).to(center.device)
        return overall_mask

    def _mask_center_block(self, center, noaug=False):
        """Block masking - mask spatially contiguous groups."""
        B, G, _ = center.shape
        
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)

        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx), device=center.device)
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx)
        return bool_masked_pos

    def forward(self, neighborhood, center, noaug=False):
        # Generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        # Encode groups
        group_input_tokens = self.encoder(neighborhood)
        
        batch_size, seq_len, C = group_input_tokens.size()

        # Get visible tokens
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        
        # Position embedding for visible tokens
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # Transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos
