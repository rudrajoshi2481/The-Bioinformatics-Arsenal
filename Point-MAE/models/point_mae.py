"""
Point-MAE: Masked Autoencoder for Point Cloud Self-supervised Learning.

Main model class that combines all components.
"""

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from .encoder import Group
from .transformer import MaskTransformer, TransformerDecoder
from .utils import trunc_normal_


class PointMAE(nn.Module):
    """Point-MAE: Masked Autoencoder for Point Cloud Self-supervised Learning."""
    
    def __init__(self, num_group=64, group_size=32, trans_dim=384, depth=12, 
                 decoder_depth=4, num_heads=6, decoder_num_heads=6, encoder_dims=384,
                 input_channel=6, mask_ratio=0.6, mask_type='rand', drop_path_rate=0.1):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.input_channel = input_channel
        self.num_group = num_group
        self.group_size = group_size
        
        # Grouping
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        
        # MAE Encoder
        self.MAE_encoder = MaskTransformer(
            trans_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            encoder_dims=encoder_dims,
            input_channel=input_channel,
            mask_ratio=mask_ratio,
            mask_type=mask_type,
            drop_path_rate=drop_path_rate,
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        trunc_normal_(self.mask_token, std=.02)
        
        # Decoder position embedding
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )
        
        # MAE Decoder
        self.MAE_decoder = TransformerDecoder(
            embed_dim=trans_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            drop_path_rate=drop_path_rate,
        )
        
        # Prediction head - outputs xyz only (3 channels)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(trans_dim, 3 * group_size, 1)
        )
        
        print(f"[PointMAE] Initialized:")
        print(f"  - Groups: {num_group} x {group_size} points")
        print(f"  - Input channels: {input_channel}")
        print(f"  - Trans dim: {trans_dim}")
        print(f"  - Mask ratio: {mask_ratio}")

    def forward(self, pts, vis=False):
        """
        Args:
            pts: (B, N, C) input point cloud
            vis: if True, return visualization outputs
            
        Returns:
            loss: reconstruction loss (if vis=False)
            (full_rec, vis_pts, centers): visualization outputs (if vis=True)
        """
        # Group points
        neighborhood, center = self.group_divider(pts)
        
        # Encode visible tokens
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape
        
        # Position embeddings
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        
        _, N, _ = pos_emd_mask.shape
        
        # Expand mask tokens
        mask_token = self.mask_token.expand(B, N, -1)
        
        # Concatenate visible and mask tokens
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        
        # Decode
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        
        B, M, C = x_rec.shape
        
        # Predict xyz
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)
        rebuild_points = rebuild_points.reshape(B * M, -1, 3)
        
        # Ground truth (xyz only from masked groups)
        gt_points = neighborhood[mask].reshape(B * M, -1, self.input_channel)[:, :, :3]
        
        # Chamfer distance loss
        loss, _ = chamfer_distance(rebuild_points, gt_points, 
                                   batch_reduction='mean', point_reduction='mean')
        
        if vis:
            # Visualization mode
            num_vis = self.num_group - M
            
            # Visible points
            vis_neighborhood = neighborhood[~mask].reshape(B, num_vis, self.group_size, self.input_channel)
            vis_xyz = vis_neighborhood[:, :, :, :3]
            
            # Add center back to visible points
            vis_center = center[~mask].reshape(B, num_vis, 3)
            full_vis = vis_xyz + vis_center.unsqueeze(2)
            
            # Reconstructed points: add center back
            rebuild_xyz = rebuild_points.reshape(B, M, self.group_size, 3)
            mask_center = center[mask].reshape(B, M, 3)
            full_rebuild = rebuild_xyz + mask_center.unsqueeze(2)
            
            # Flatten and combine
            full_vis_flat = full_vis.reshape(B, -1, 3)
            full_rebuild_flat = full_rebuild.reshape(B, -1, 3)
            
            full_rec = torch.cat([full_vis_flat, full_rebuild_flat], dim=1)
            vis_pts = full_vis_flat
            
            return full_rec, vis_pts, center
        
        return loss
