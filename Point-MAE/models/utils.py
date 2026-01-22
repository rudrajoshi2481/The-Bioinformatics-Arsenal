"""
Utility functions for Point-MAE.

Contains:
- trunc_normal_: Truncated normal initialization
- fps_cpu: Farthest Point Sampling (CPU version)
- knn_cpu: K-Nearest Neighbors (CPU version)
"""

import torch
import numpy as np


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        l = (1. + torch.erf(torch.tensor((a - mean) / std / np.sqrt(2.)))) / 2.
        u = (1. + torch.erf(torch.tensor((b - mean) / std / np.sqrt(2.)))) / 2.
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * np.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def fps_cpu(xyz, npoint):
    """
    Farthest Point Sampling (CPU version).
    
    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample
        
    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Random starting point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids


def knn_cpu(x, y, k):
    """
    K-Nearest Neighbors (CPU version).
    
    Args:
        x: (B, N, 3) query points
        y: (B, M, 3) reference points
        k: number of neighbors
        
    Returns:
        idx: (B, N, k) indices of k nearest neighbors in y for each point in x
    """
    dist = torch.cdist(x, y)
    _, idx = torch.topk(dist, k, dim=-1, largest=False)
    return idx
