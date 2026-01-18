"""
Hi-C Enhancement Dataset
========================

Dataset classes for Hi-C enhancement training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple


class HiCDataset(Dataset):
    """Dataset for Hi-C enhancement training"""
    
    def __init__(self, npz_file: str, augment: bool = False):
        """
        Args:
            npz_file: Path to .npz file with 'data' and 'target' arrays
            augment: Whether to apply data augmentation
        """
        data = np.load(npz_file)
        self.low_res = torch.from_numpy(data['data']).float()
        self.high_res = torch.from_numpy(data['target']).float()
        self.augment = augment
        
        print(f"Loaded dataset: {len(self.low_res)} samples")
        print(f"  Input shape: {self.low_res.shape}")
        print(f"  Target shape: {self.high_res.shape}")
    
    def __len__(self) -> int:
        return len(self.low_res)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low = self.low_res[idx]
        high = self.high_res[idx]
        
        if self.augment and np.random.rand() > 0.5:
            # Random flip
            if np.random.rand() > 0.5:
                low = torch.flip(low, [1])
                high = torch.flip(high, [1])
            if np.random.rand() > 0.5:
                low = torch.flip(low, [2])
                high = torch.flip(high, [2])
            
            # Random transpose (Hi-C symmetry)
            if np.random.rand() > 0.5:
                low = low.transpose(1, 2)
                high = high.transpose(1, 2)
        
        return low, high


class HiCInferenceDataset(Dataset):
    """Dataset for inference on full chromosomes"""
    
    def __init__(self, chunks: np.ndarray, indices: np.ndarray):
        """
        Args:
            chunks: Array of shape (N, 1, H, W) containing input chunks
            indices: Array of shape (N, 4) with (chr, size, i, j) indices
        """
        self.chunks = torch.from_numpy(chunks).float()
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        return self.chunks[idx], self.indices[idx]


def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders
    
    Args:
        train_path: Path to training data .npz file
        val_path: Path to validation data .npz file (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to augment training data
    
    Returns:
        train_loader, val_loader (or None if no val_path)
    """
    train_dataset = HiCDataset(train_path, augment=augment)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = HiCDataset(val_path, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


def split_dataset(
    npz_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split a single dataset into train/val/test sets
    
    Args:
        npz_file: Path to input .npz file
        output_dir: Directory to save split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        Paths to train, val, test .npz files
    """
    np.random.seed(seed)
    
    data = np.load(npz_file)
    low_res = data['data']
    high_res = data['target']
    indices = data.get('indices', np.arange(len(low_res)))
    
    n_samples = len(low_res)
    perm = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_path = f"{output_dir}/train_dataset.npz"
    val_path = f"{output_dir}/val_dataset.npz"
    test_path = f"{output_dir}/test_dataset.npz"
    
    np.savez_compressed(train_path, data=low_res[train_idx], target=high_res[train_idx])
    np.savez_compressed(val_path, data=low_res[val_idx], target=high_res[val_idx])
    np.savez_compressed(test_path, data=low_res[test_idx], target=high_res[test_idx])
    
    print(f"Split dataset: {n_samples} samples")
    print(f"  Train: {len(train_idx)} samples -> {train_path}")
    print(f"  Val: {len(val_idx)} samples -> {val_path}")
    print(f"  Test: {len(test_idx)} samples -> {test_path}")
    
    return train_path, val_path, test_path
