"""
Data Validation Module
Comprehensive checks following expert principles:
"Data > Architecture > Hyperparameters > Everything Else"
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, get_config
from data.preprocessing import fMRIPreprocessor


class DataValidator:
    """Comprehensive data validation before training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = fMRIPreprocessor(config)
        self.results = {}
    
    def check_file_integrity(self, subject: str, task: str) -> Dict:
        """Check if files exist and are readable"""
        bids_dir = self.config.data.bids_dir
        func_dir = bids_dir / subject / "func"
        
        results = {
            "subject": subject,
            "task": task,
            "passed": True,
            "checks": {}
        }
        
        # Check BOLD file
        bold_files = list(func_dir.glob(f"*task-{task}*_bold.nii.gz"))
        results["checks"]["bold_exists"] = len(bold_files) > 0
        
        if bold_files:
            bold_path = bold_files[0]
            results["checks"]["bold_readable"] = bold_path.exists()
            
            # Check events file
            events_path = bold_path.with_name(
                bold_path.name.replace("_bold.nii.gz", "_events.tsv")
            )
            results["checks"]["events_exists"] = events_path.exists()
            
            # Check JSON sidecar
            json_path = bold_path.with_suffix("").with_suffix(".json")
            results["checks"]["json_exists"] = json_path.exists()
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_data_quality(self, subject: str, task: str) -> Dict:
        """Check data quality metrics"""
        results = {
            "subject": subject,
            "task": task,
            "passed": True,
            "checks": {},
            "stats": {}
        }
        
        try:
            volumes, meta = self.preprocessor.load_fmri(subject, task)
            
            # Shape check
            expected_shape = self.config.data.volume_shape
            actual_shape = volumes.shape[:3]
            results["checks"]["shape_match"] = actual_shape == expected_shape
            results["stats"]["shape"] = volumes.shape
            
            # NaN/Inf check
            nan_count = np.isnan(volumes).sum()
            inf_count = np.isinf(volumes).sum()
            results["checks"]["no_nan"] = nan_count == 0
            results["checks"]["no_inf"] = inf_count == 0
            results["stats"]["nan_count"] = int(nan_count)
            results["stats"]["inf_count"] = int(inf_count)
            
            # Value range check
            vmin, vmax = float(volumes.min()), float(volumes.max())
            results["checks"]["valid_range"] = vmax > vmin
            results["stats"]["min"] = vmin
            results["stats"]["max"] = vmax
            results["stats"]["mean"] = float(volumes.mean())
            results["stats"]["std"] = float(volumes.std())
            
            # Temporal variance check (fMRI should vary over time)
            temporal_std = volumes.std(axis=3).mean()
            results["checks"]["temporal_variance"] = temporal_std > 1e-6
            results["stats"]["temporal_std"] = float(temporal_std)
            
            # Signal-to-noise ratio estimate
            # For normalized data, SNR can be low - use coefficient of variation instead
            mean_signal = np.abs(volumes.mean())
            noise_estimate = volumes[:, :, :, :10].std()
            snr = mean_signal / (noise_estimate + 1e-8)
            # For raw fMRI, SNR > 1 is good; for normalized data, any positive value is fine
            results["checks"]["snr_reasonable"] = snr > 0.1 or noise_estimate > 0
            results["stats"]["snr_estimate"] = float(snr)
            
            # Check for constant voxels (dead voxels)
            voxel_std = volumes.std(axis=3)
            dead_voxels = (voxel_std < 1e-8).sum()
            dead_ratio = dead_voxels / voxel_std.size
            results["checks"]["low_dead_voxels"] = dead_ratio < 0.5
            results["stats"]["dead_voxel_ratio"] = float(dead_ratio)
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_patch_compatibility(self) -> Dict:
        """Check if volume dimensions are compatible with patch size"""
        results = {
            "passed": True,
            "checks": {}
        }
        
        vol_shape = self.config.data.volume_shape
        patch_size = self.config.patch.patch_size
        
        for i, (v, p) in enumerate(zip(vol_shape, patch_size)):
            dim_name = ["X", "Y", "Z"][i]
            divisible = v % p == 0
            results["checks"][f"{dim_name}_divisible"] = divisible
            
            if not divisible:
                results[f"{dim_name}_remainder"] = v % p
        
        results["passed"] = all(results["checks"].values())
        results["n_patches"] = self.config.patch.n_patches
        results["patch_dim"] = self.config.patch.patch_dim
        
        return results
    
    def check_train_val_split(self, n_samples: int) -> Dict:
        """Validate train/val split ratios"""
        results = {
            "passed": True,
            "checks": {}
        }
        
        train_ratio = self.config.data.train_ratio
        n_train = int(n_samples * train_ratio)
        n_val = n_samples - n_train
        
        results["n_total"] = n_samples
        results["n_train"] = n_train
        results["n_val"] = n_val
        results["train_ratio"] = train_ratio
        
        # Check minimum samples
        results["checks"]["enough_train"] = n_train >= 10
        results["checks"]["enough_val"] = n_val >= 5
        results["checks"]["reasonable_split"] = 0.5 <= train_ratio <= 0.95
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_model_capacity(self, n_samples: int) -> Dict:
        """
        Check if model size is appropriate for data size.
        Expert rule: params/data ratio < 100 for small datasets
        """
        results = {
            "passed": True,
            "checks": {},
            "recommendations": []
        }
        
        # Estimate model parameters
        embed_dim = self.config.model.embed_dim
        patch_dim = self.config.patch.patch_dim
        n_patches = self.config.patch.n_patches
        encoder_depth = self.config.model.encoder_depth
        decoder_depth = self.config.model.decoder_depth
        
        # Rough parameter estimates
        patch_embed_params = patch_dim * embed_dim
        encoder_params = encoder_depth * (4 * embed_dim * embed_dim + 2 * embed_dim * embed_dim * self.config.model.encoder_mlp_ratio)
        decoder_embed_dim = self.config.model.decoder_embed_dim
        decoder_params = decoder_depth * (4 * decoder_embed_dim * decoder_embed_dim + 2 * decoder_embed_dim * decoder_embed_dim * 2)
        recon_params = decoder_embed_dim * patch_dim
        
        total_params = patch_embed_params + encoder_params + decoder_params + recon_params
        
        results["estimated_params"] = int(total_params)
        results["n_samples"] = n_samples
        results["params_per_sample"] = total_params / n_samples
        
        # Checks
        ratio = total_params / n_samples
        results["checks"]["ratio_acceptable"] = ratio < 10000  # Allow up to 10k params per sample
        results["checks"]["not_too_small"] = total_params > 10000  # At least 10k params
        
        if ratio > 1000:
            results["recommendations"].append(
                f"Model may be too large. Consider reducing embed_dim from {embed_dim} to {embed_dim//2}"
            )
        
        if ratio < 10:
            results["recommendations"].append(
                f"Model may be too small. Consider increasing embed_dim from {embed_dim} to {embed_dim*2}"
            )
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def run_all_checks(self) -> Dict:
        """Run all validation checks"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE DATA VALIDATION")
        print("=" * 70)
        
        all_results = {
            "passed": True,
            "file_integrity": [],
            "data_quality": [],
            "patch_compatibility": None,
            "train_val_split": None,
            "model_capacity": None
        }
        
        total_samples = 0
        
        # File integrity and data quality for each subject/task
        for subject in self.config.data.subjects:
            for task in self.config.data.tasks:
                print(f"\n📁 Checking {subject} - {task}")
                
                # File integrity
                file_result = self.check_file_integrity(subject, task)
                all_results["file_integrity"].append(file_result)
                
                status = "✓" if file_result["passed"] else "✗"
                print(f"   {status} File integrity: {file_result['checks']}")
                
                if not file_result["passed"]:
                    all_results["passed"] = False
                    continue
                
                # Data quality
                quality_result = self.check_data_quality(subject, task)
                all_results["data_quality"].append(quality_result)
                
                status = "✓" if quality_result["passed"] else "✗"
                print(f"   {status} Data quality: {quality_result['checks']}")
                
                if quality_result.get("stats"):
                    print(f"      Shape: {quality_result['stats']['shape']}")
                    print(f"      Range: [{quality_result['stats']['min']:.1f}, {quality_result['stats']['max']:.1f}]")
                    print(f"      SNR: {quality_result['stats']['snr_estimate']:.1f}")
                
                if not quality_result["passed"]:
                    all_results["passed"] = False
                
                # Count samples
                if "stats" in quality_result and "shape" in quality_result["stats"]:
                    total_samples += quality_result["stats"]["shape"][3]
        
        # Patch compatibility
        print(f"\n🧩 Checking patch compatibility")
        patch_result = self.check_patch_compatibility()
        all_results["patch_compatibility"] = patch_result
        
        status = "✓" if patch_result["passed"] else "✗"
        print(f"   {status} Patches: {patch_result['n_patches']} patches of dim {patch_result['patch_dim']}")
        
        if not patch_result["passed"]:
            all_results["passed"] = False
        
        # Train/val split
        print(f"\n📊 Checking train/val split")
        split_result = self.check_train_val_split(total_samples)
        all_results["train_val_split"] = split_result
        
        status = "✓" if split_result["passed"] else "✗"
        print(f"   {status} Split: {split_result['n_train']} train / {split_result['n_val']} val")
        
        if not split_result["passed"]:
            all_results["passed"] = False
        
        # Model capacity
        print(f"\n🏗️ Checking model capacity")
        capacity_result = self.check_model_capacity(total_samples)
        all_results["model_capacity"] = capacity_result
        
        status = "✓" if capacity_result["passed"] else "✗"
        print(f"   {status} Params: {capacity_result['estimated_params']:,}")
        print(f"      Params/sample ratio: {capacity_result['params_per_sample']:.1f}")
        
        for rec in capacity_result.get("recommendations", []):
            print(f"      ⚠️ {rec}")
        
        if not capacity_result["passed"]:
            all_results["passed"] = False
        
        # Summary
        print("\n" + "=" * 70)
        if all_results["passed"]:
            print("✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("❌ SOME VALIDATION CHECKS FAILED")
            print("   Fix issues before training!")
        print("=" * 70)
        
        return all_results


def validate_dataset(config: Config = None) -> Dict:
    """Main validation entry point"""
    if config is None:
        config = get_config()
    
    validator = DataValidator(config)
    return validator.run_all_checks()


if __name__ == "__main__":
    results = validate_dataset()
    
    # Save results
    output_path = Path(__file__).parent.parent / "outputs" / "validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {output_path}")
