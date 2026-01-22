"""
Dataset loaders for Point-MAE.

Supports:
- ModelNet40 .dat files (preprocessed with FPS)
- ModelNet40 raw .txt files
"""

import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if max_dist > 0:
        pc = pc / max_dist
    return pc


class ModelNet40Dataset(Dataset):
    """ModelNet40 dataset loader for .dat files.
    
    Expects .dat files with format: modelnet40_{split}_8192pts_fps.dat
    Each .dat file contains a list of (point_cloud, label) tuples.
    """
    
    def __init__(self, root, npoints=1024, split='train', use_normals=True):
        """
        Args:
            root: Directory containing .dat files
            npoints: Number of points to sample
            split: 'train' or 'test'
            use_normals: Use 6 channels (xyz+normals) vs 3 (xyz only)
        """
        self.root = root
        self.npoints = npoints
        self.split = split
        self.use_normals = use_normals
        
        dat_file = os.path.join(root, f'modelnet40_{split}_8192pts_fps.dat')
        
        if not os.path.exists(dat_file):
            raise FileNotFoundError(
                f"Dataset not found: {dat_file}\n"
                f"Please run preprocessing first. See README.md for instructions."
            )
        
        print(f'Loading {split} data from {dat_file}...')
        with open(dat_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.data = self._parse_data(raw_data)
        print(f'Loaded {len(self.data)} samples')
    
    def _parse_data(self, raw_data):
        """Parse raw data into list of (points, label) tuples."""
        parsed = []
        
        def is_valid(pts):
            if pts is None:
                return False
            pts = np.array(pts)
            return pts.ndim == 2 and pts.shape[0] >= 100 and pts.shape[1] in [3, 6]
        
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first = raw_data[0]
            
            if isinstance(first, (list, tuple)) and len(first) == 2:
                inner = np.array(first[0])
                if inner.ndim == 2 and inner.shape[1] in [3, 6]:
                    for pts, label in raw_data:
                        pts = np.array(pts)
                        if is_valid(pts):
                            parsed.append((pts, label))
                else:
                    for item in raw_data:
                        if isinstance(item, (list, tuple)):
                            for pts in item:
                                pts = np.array(pts)
                                if is_valid(pts):
                                    parsed.append((pts, len(parsed) % 40))
            elif hasattr(first, 'shape') and first.ndim == 2:
                for i, pts in enumerate(raw_data):
                    if is_valid(pts):
                        parsed.append((pts, i % 40))
        
        if len(parsed) == 0:
            raise ValueError("Could not parse dataset - no valid point clouds found")
        
        return parsed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pts, label = self.data[idx]
        pts = np.array(pts).copy()
        
        if pts.shape[0] > self.npoints:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=False)
            pts = pts[choice]
        elif pts.shape[0] < self.npoints:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
            pts = pts[choice]
        
        pts[:, :3] = pc_normalize(pts[:, :3])
        
        if not self.use_normals and pts.shape[1] > 3:
            pts = pts[:, :3]
        
        return torch.from_numpy(pts).float(), label
