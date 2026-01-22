"""
Point Cloud Encoder for Point-MAE.

Contains:
- Encoder: Conv1D-based point cloud group encoder
- Group: FPS + KNN grouping module
"""

import torch
import torch.nn as nn
from .utils import fps_cpu, knn_cpu


class Encoder(nn.Module):
    """Point cloud group encoder using Conv1D."""
    
    def __init__(self, encoder_channel, input_channel=6):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.input_channel = input_channel
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, N, C) grouped point clouds
            
        Returns:
            features: (B, G, encoder_channel)
        """
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS + KNN grouping module."""
    
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, pts):
        """
        Args:
            pts: (B, N, C) input points (C=6 for xyz+normals, C=3 for xyz only)
            
        Returns:
            neighborhood: (B, G, M, C) grouped points (normalized xyz)
            center: (B, G, 3) group centers
        """
        batch_size, num_points, num_channels = pts.shape
        xyz = pts[:, :, :3].contiguous()
        
        # FPS to get centers
        fps_idx = fps_cpu(xyz, self.num_group)
        center = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
        
        # KNN to get neighborhoods
        idx = knn_cpu(center, xyz, self.group_size)
        
        # Gather points
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, num_channels)
        pts_expanded = pts.unsqueeze(1).expand(-1, self.num_group, -1, -1)
        neighborhood = torch.gather(pts_expanded, 2, idx_expanded)
        
        # Normalize xyz (subtract center)
        neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] - center.unsqueeze(2)
        
        return neighborhood, center
