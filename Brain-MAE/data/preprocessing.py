"""
fMRI Data Preprocessing Module
Handles loading, normalization, patching, and wavelet transforms.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, get_config


class fMRIPreprocessor:
    """Preprocesses fMRI volumes for MAE training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.wavelet_enabled = config.wavelet.enabled
        
        if self.wavelet_enabled:
            try:
                import pywt
                self.pywt = pywt
            except ImportError:
                print("Warning: pywt not installed. Disabling wavelet transform.")
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
        json_path = bold_path.with_suffix("").with_suffix(".json")
        
        # Load NIfTI
        img = nib.load(bold_path)
        volumes = img.get_fdata()
        
        # Get TR from header or JSON
        tr = img.header.get_zooms()[3]
        
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
    
    def apply_wavelet_transform(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply 3D wavelet transform to a single volume.
        
        Args:
            volume: (X, Y, Z) array
        
        Returns:
            Dictionary with wavelet coefficients
        """
        if not self.wavelet_enabled:
            return {"raw": volume}
        
        coeffs = self.pywt.dwtn(volume, self.config.wavelet.wavelet)
        
        # coeffs is a dict with keys like 'aaa', 'aad', 'ada', etc.
        # 'a' = approximation, 'd' = detail
        # For 3D: 8 subbands (2^3)
        return coeffs
    
    def inverse_wavelet_transform(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """Inverse 3D wavelet transform"""
        if not self.wavelet_enabled or "raw" in coeffs:
            return coeffs.get("raw", list(coeffs.values())[0])
        
        return self.pywt.idwtn(coeffs, self.config.wavelet.wavelet)
    
    def extract_patches(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract 3D patches from a volume.
        
        Args:
            volume: (X, Y, Z) array
        
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
        
        return np.array(patches)  # (n_patches, patch_dim)
    
    def reconstruct_from_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Reconstruct volume from patches.
        
        Args:
            patches: (n_patches, patch_dim) array
        
        Returns:
            volume: (X, Y, Z) array
        """
        px, py, pz = self.config.patch.patch_size
        gx, gy, gz = self.config.patch.grid_size
        X, Y, Z = self.config.data.volume_shape
        
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
    """PyTorch Dataset for fMRI volumes"""
    
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
        
        subjects = subjects or config.data.subjects
        tasks = tasks or config.data.tasks
        
        # Load and preprocess all data
        self.volumes = []
        self.metadata = []
        
        for subject in subjects:
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
        n_train = int(n_total * config.data.train_ratio)
        
        # Shuffle indices
        np.random.seed(config.data.random_seed)
        indices = np.random.permutation(n_total)
        
        if split == "train":
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]
        
        print(f"{split.upper()} set: {len(self.indices)} volumes")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        volume = self.volumes[real_idx]
        
        # Extract patches
        patches = self.preprocessor.extract_patches(volume)
        
        # Convert to tensor
        patches_tensor = torch.from_numpy(patches).float()
        volume_tensor = torch.from_numpy(volume).float()
        
        return {
            "patches": patches_tensor,  # (n_patches, patch_dim)
            "volume": volume_tensor,    # (X, Y, Z)
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


def run_data_quality_checks(config: Config) -> Dict:
    """
    Run data quality checks before training.
    Expert principle: Always validate data first!
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    results = {
        "passed": True,
        "checks": {}
    }
    
    preprocessor = fMRIPreprocessor(config)
    
    for subject in config.data.subjects:
        for task in config.data.tasks:
            print(f"\n📊 Checking {subject} - {task}")
            
            try:
                volumes, meta = preprocessor.load_fmri(subject, task)
                
                # Check 1: Shape
                expected_shape = (*config.data.volume_shape, meta["n_volumes"])
                shape_ok = volumes.shape[:3] == config.data.volume_shape
                print(f"  ✓ Shape: {volumes.shape}" if shape_ok else f"  ✗ Shape mismatch: {volumes.shape}")
                
                # Check 2: Data range
                vmin, vmax = volumes.min(), volumes.max()
                range_ok = vmax > vmin and not np.isnan(volumes).any()
                print(f"  ✓ Range: [{vmin:.1f}, {vmax:.1f}]" if range_ok else f"  ✗ Invalid range")
                
                # Check 3: No NaN/Inf
                nan_count = np.isnan(volumes).sum()
                inf_count = np.isinf(volumes).sum()
                clean_ok = nan_count == 0 and inf_count == 0
                print(f"  ✓ Clean data (no NaN/Inf)" if clean_ok else f"  ✗ Found {nan_count} NaN, {inf_count} Inf")
                
                # Check 4: Signal variance
                mean_signal = volumes.mean()
                std_signal = volumes.std()
                variance_ok = std_signal > 1e-6
                print(f"  ✓ Signal: mean={mean_signal:.1f}, std={std_signal:.1f}" if variance_ok else f"  ✗ No variance")
                
                # Check 5: Temporal variance (fMRI should vary over time)
                temporal_std = volumes.std(axis=3).mean()
                temporal_ok = temporal_std > 1e-6
                print(f"  ✓ Temporal variance: {temporal_std:.2f}" if temporal_ok else f"  ✗ No temporal variance")
                
                all_ok = shape_ok and range_ok and clean_ok and variance_ok and temporal_ok
                results["checks"][f"{subject}_{task}"] = all_ok
                
                if not all_ok:
                    results["passed"] = False
                    
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                results["checks"][f"{subject}_{task}"] = False
                results["passed"] = False
    
    # Summary
    print("\n" + "-" * 60)
    if results["passed"]:
        print("✅ All data quality checks PASSED")
    else:
        print("❌ Some data quality checks FAILED")
        print("   Fix data issues before training!")
    
    return results


if __name__ == "__main__":
    # Test preprocessing
    config = get_config()
    
    # Run quality checks
    results = run_data_quality_checks(config)
    
    if results["passed"]:
        # Test dataset creation
        print("\n" + "=" * 60)
        print("TESTING DATASET")
        print("=" * 60)
        
        train_loader, val_loader = create_dataloaders(config)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  patches: {batch['patches'].shape}")
        print(f"  volume: {batch['volume'].shape}")
        
        # Test patch reconstruction
        preprocessor = fMRIPreprocessor(config)
        patches_np = batch['patches'][0].numpy()
        reconstructed = preprocessor.reconstruct_from_patches(patches_np)
        original = batch['volume'][0].numpy()
        
        recon_error = np.abs(reconstructed - original).mean()
        print(f"\nPatch reconstruction error: {recon_error:.6f}")
        print("✓ Patches can be reconstructed perfectly" if recon_error < 1e-6 else "✗ Reconstruction error!")
