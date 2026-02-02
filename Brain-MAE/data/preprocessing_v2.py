"""
fMRI Data Preprocessing Module - Production Version
Handles loading, normalization, padding, and patching for multi-subject training.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, get_config, get_prototype_config


class fMRIPreprocessor:
    """Preprocesses fMRI volumes for MAE training - Production version"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Padding configuration
        self.pad_depth = config.patch.pad_depth
        self.original_shape = config.data.volume_shape
        self.padded_shape = config.patch.padded_shape if self.pad_depth else self.original_shape
        
        # Wavelet is disabled in production config
        self.wavelet_enabled = False
    
    def load_fmri(self, subject: str, task: str) -> Tuple[np.ndarray, dict]:
        """
        Load fMRI data for a subject/task.
        
        Returns:
            volumes: (X, Y, Z, T) array
            metadata: dict with TR, shape info
        """
        bids_dir = self.config.data.bids_dir
        
        # Find the BOLD file
        func_dir = bids_dir / subject / "func"
        bold_files = list(func_dir.glob(f"*task-{task}*_bold.nii.gz"))
        
        if not bold_files:
            raise FileNotFoundError(f"No BOLD file found for {subject}, task {task}")
        
        bold_path = bold_files[0]
        
        # Load NIfTI
        img = nib.load(bold_path)
        volumes = img.get_fdata().astype(np.float32)
        
        # Get TR from header
        tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 1.5
        
        metadata = {
            "path": str(bold_path),
            "shape": volumes.shape,
            "tr": tr,
            "n_volumes": volumes.shape[3],
            "voxel_size": img.header.get_zooms()[:3]
        }
        
        print(f"Loaded {subject} {task}: shape={volumes.shape}, TR={tr:.2f}s")
        return volumes, metadata
    
    def normalize_volumes(self, volumes: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize fMRI volumes.
        
        Args:
            volumes: (X, Y, Z, T) array
            method: 'zscore' (per volume), 'global', or 'minmax'
        
        Returns:
            Normalized volumes
        """
        volumes = volumes.astype(np.float32)
        
        if method == "zscore":
            # Z-score normalize each volume independently
            for t in range(volumes.shape[3]):
                vol = volumes[:, :, :, t]
                mean = vol.mean()
                std = vol.std()
                if std > 1e-8:
                    volumes[:, :, :, t] = (vol - mean) / std
                else:
                    volumes[:, :, :, t] = vol - mean
                    
        elif method == "global":
            # Global z-score across all volumes
            mean = volumes.mean()
            std = volumes.std()
            volumes = (volumes - mean) / (std + 1e-8)
            
        elif method == "minmax":
            # Min-max to [0, 1]
            vmin, vmax = volumes.min(), volumes.max()
            volumes = (volumes - vmin) / (vmax - vmin + 1e-8)
        
        return volumes
    
    def pad_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Pad volume to make dimensions divisible by patch size.
        
        Args:
            volume: (X, Y, Z) array with original shape
        
        Returns:
            Padded volume with shape matching padded_shape
        """
        if not self.pad_depth:
            return volume
        
        orig_shape = volume.shape
        target_shape = self.padded_shape
        
        # Calculate padding needed
        pad_x = target_shape[0] - orig_shape[0]
        pad_y = target_shape[1] - orig_shape[1]
        pad_z = target_shape[2] - orig_shape[2]
        
        # Pad symmetrically (or asymmetrically if odd)
        pad_width = (
            (pad_x // 2, pad_x - pad_x // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_z // 2, pad_z - pad_z // 2)
        )
        
        # Use edge padding to avoid artifacts
        padded = np.pad(volume, pad_width, mode='edge')
        
        return padded
    
    def unpad_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Remove padding from volume.
        
        Args:
            volume: (X, Y, Z) padded array
        
        Returns:
            Original shape volume
        """
        if not self.pad_depth:
            return volume
        
        orig_shape = self.original_shape
        padded_shape = self.padded_shape
        
        # Calculate padding that was added
        pad_x = padded_shape[0] - orig_shape[0]
        pad_y = padded_shape[1] - orig_shape[1]
        pad_z = padded_shape[2] - orig_shape[2]
        
        # Extract original region
        start_x = pad_x // 2
        start_y = pad_y // 2
        start_z = pad_z // 2
        
        unpadded = volume[
            start_x:start_x + orig_shape[0],
            start_y:start_y + orig_shape[1],
            start_z:start_z + orig_shape[2]
        ]
        
        return unpadded
    
    def extract_patches(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract 3D patches from a volume.
        
        Args:
            volume: (X, Y, Z) array (should be padded if pad_depth=True)
        
        Returns:
            patches: (n_patches, patch_dim) array
        """
        px, py, pz = self.config.patch.patch_size
        gx, gy, gz = self.config.patch.grid_size
        
        patches = []
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    patch = volume[
                        i*px:(i+1)*px,
                        j*py:(j+1)*py,
                        k*pz:(k+1)*pz
                    ]
                    patches.append(patch.flatten())
        
        return np.array(patches, dtype=np.float32)  # (n_patches, patch_dim)
    
    def reconstruct_from_patches(self, patches: np.ndarray, unpad: bool = True) -> np.ndarray:
        """
        Reconstruct volume from patches.
        
        Args:
            patches: (n_patches, patch_dim) array
            unpad: If True, remove padding to return original shape
        
        Returns:
            volume: (X, Y, Z) array
        """
        px, py, pz = self.config.patch.patch_size
        gx, gy, gz = self.config.patch.grid_size
        
        # Reconstruct to padded shape
        X, Y, Z = self.padded_shape if self.pad_depth else self.original_shape
        
        volume = np.zeros((X, Y, Z), dtype=np.float32)
        
        idx = 0
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    patch = patches[idx].reshape(px, py, pz)
                    volume[
                        i*px:(i+1)*px,
                        j*py:(j+1)*py,
                        k*pz:(k+1)*pz
                    ] = patch
                    idx += 1
        
        if unpad and self.pad_depth:
            volume = self.unpad_volume(volume)
        
        return volume
    
    def get_patch_positions(self) -> np.ndarray:
        """
        Get 3D positions for each patch (for positional encoding).
        
        Returns:
            positions: (n_patches, 3) array with (i, j, k) grid positions
        """
        gx, gy, gz = self.config.patch.grid_size
        positions = []
        
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    positions.append([i, j, k])
        
        return np.array(positions)


class fMRIDataset(Dataset):
    """PyTorch Dataset for fMRI volumes - Production version"""
    
    def __init__(
        self,
        config: Config,
        subjects: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        split: str = "train"
    ):
        self.config = config
        self.preprocessor = fMRIPreprocessor(config)
        self.split = split
        
        # Get subjects - either provided or from config
        if subjects:
            self.subjects = subjects
        elif config.data.subjects:
            self.subjects = config.data.subjects
        else:
            # Auto-discover subjects from BIDS directory
            self.subjects = self._discover_subjects()
        
        tasks = tasks or config.data.tasks
        
        # Load and preprocess all data
        self.volumes = []
        self.metadata = []
        
        for subject in self.subjects:
            for task in tasks:
                try:
                    vols, meta = self.preprocessor.load_fmri(subject, task)
                    vols = self.preprocessor.normalize_volumes(vols)
                    
                    # Store each volume separately
                    for t in range(vols.shape[3]):
                        self.volumes.append(vols[:, :, :, t])
                        self.metadata.append({
                            "subject": subject,
                            "task": task,
                            "volume_idx": t,
                            **meta
                        })
                except Exception as e:
                    print(f"Warning: Could not load {subject} {task}: {e}")
        
        print(f"Loaded {len(self.volumes)} volumes total")
        
        # Train/val split
        n_total = len(self.volumes)
        n_train = int(n_total * config.data.train_subjects_ratio)
        n_val = int(n_total * config.data.val_subjects_ratio)
        
        # Shuffle indices
        np.random.seed(config.data.random_seed)
        indices = np.random.permutation(n_total)
        
        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train:n_train + n_val]
        else:  # test
            self.indices = indices[n_train + n_val:]
        
        print(f"{split.upper()} set: {len(self.indices)} volumes")
    
    def _discover_subjects(self) -> List[str]:
        """Auto-discover subjects from BIDS directory"""
        bids_dir = self.config.data.bids_dir
        subjects = []
        for d in sorted(bids_dir.iterdir()):
            if d.is_dir() and d.name.startswith("sub-"):
                subjects.append(d.name)
        return subjects[:10]  # Limit for safety
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        volume = self.volumes[real_idx]
        
        # Pad volume if needed
        if self.preprocessor.pad_depth:
            volume = self.preprocessor.pad_volume(volume)
        
        # Extract patches
        patches = self.preprocessor.extract_patches(volume)
        
        # Convert to tensor
        patches_tensor = torch.from_numpy(patches).float()
        volume_tensor = torch.from_numpy(volume).float()
        
        return {
            "patches": patches_tensor,  # (n_patches, patch_dim)
            "volume": volume_tensor,    # (X, Y, Z) - padded if applicable
            "idx": real_idx
        }


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = fMRIDataset(config, split="train")
    val_dataset = fMRIDataset(config, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.device == "cuda"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.device == "cuda"
    )
    
    return train_loader, val_loader


def test_patch_reconstruction():
    """Test that patches can be perfectly reconstructed"""
    print("\n" + "=" * 60)
    print("TESTING PATCH RECONSTRUCTION")
    print("=" * 60)
    
    config = get_prototype_config()
    preprocessor = fMRIPreprocessor(config)
    
    # Create test volume
    original_shape = config.data.volume_shape
    test_volume = np.random.randn(*original_shape).astype(np.float32)
    
    print(f"\nOriginal volume shape: {test_volume.shape}")
    
    # Pad
    padded = preprocessor.pad_volume(test_volume)
    print(f"Padded volume shape: {padded.shape}")
    
    # Extract patches
    patches = preprocessor.extract_patches(padded)
    print(f"Patches shape: {patches.shape}")
    print(f"  -> {patches.shape[0]} patches, each {patches.shape[1]} voxels")
    
    # Reconstruct (with unpadding)
    reconstructed = preprocessor.reconstruct_from_patches(patches, unpad=True)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check error
    max_error = np.abs(reconstructed - test_volume).max()
    mean_error = np.abs(reconstructed - test_volume).mean()
    
    print(f"\nReconstruction error:")
    print(f"  Max error: {max_error:.10f}")
    print(f"  Mean error: {mean_error:.10f}")
    
    if max_error < 1e-6:
        print("✅ PERFECT RECONSTRUCTION - No information loss!")
        return True
    else:
        print("❌ ERROR: Information loss detected!")
        return False


def test_with_real_data():
    """Test with real fMRI data"""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL DATA")
    print("=" * 60)
    
    config = get_prototype_config()
    config.data.subjects = ["sub-001"]
    config.data.tasks = ["tunnel"]
    
    preprocessor = fMRIPreprocessor(config)
    
    try:
        volumes, meta = preprocessor.load_fmri("sub-001", "tunnel")
        volumes = preprocessor.normalize_volumes(volumes)
        
        # Test one volume
        test_vol = volumes[:, :, :, 0]
        print(f"\nOriginal volume shape: {test_vol.shape}")
        
        # Pad
        padded = preprocessor.pad_volume(test_vol)
        print(f"Padded volume shape: {padded.shape}")
        
        # Extract and reconstruct
        patches = preprocessor.extract_patches(padded)
        reconstructed = preprocessor.reconstruct_from_patches(patches, unpad=True)
        
        # Check
        max_error = np.abs(reconstructed - test_vol).max()
        print(f"Reconstruction error: {max_error:.10f}")
        
        if max_error < 1e-6:
            print("✅ Real data reconstruction: PERFECT")
            return True
        else:
            print("❌ Real data reconstruction: ERROR")
            return False
            
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    test1 = test_patch_reconstruction()
    test2 = test_with_real_data()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Synthetic data test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Real data test: {'✅ PASS' if test2 else '❌ FAIL'}")
