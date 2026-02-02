"""
Model Validation Module
Sanity checks and debugging tools for the Brain MAE model.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, get_config
from models.mae_model import create_model, BrainMAE, count_parameters


class ModelValidator:
    """Comprehensive model validation and debugging"""
    
    def __init__(self, config: Config, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cpu")
    
    def check_forward_pass(self, model: BrainMAE) -> Dict:
        """Test forward pass with random input"""
        results = {
            "passed": True,
            "checks": {}
        }
        
        model.eval()
        batch_size = 2
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        
        x = torch.randn(batch_size, n_patches, patch_dim).to(self.device)
        
        try:
            with torch.no_grad():
                output = model(x)
            
            # Check output shapes
            results["checks"]["recon_shape"] = output["reconstruction"].shape == x.shape
            results["checks"]["mask_shape"] = output["mask"].shape == (batch_size, n_patches)
            
            # Check for NaN/Inf
            results["checks"]["no_nan_recon"] = not torch.isnan(output["reconstruction"]).any()
            results["checks"]["no_inf_recon"] = not torch.isinf(output["reconstruction"]).any()
            
            # Check latent
            n_visible = int(n_patches * (1 - self.config.model.mask_ratio))
            expected_latent_shape = (batch_size, n_visible, self.config.model.embed_dim)
            results["checks"]["latent_shape"] = output["latent"].shape == expected_latent_shape
            
            results["output_shapes"] = {
                "reconstruction": list(output["reconstruction"].shape),
                "mask": list(output["mask"].shape),
                "latent": list(output["latent"].shape)
            }
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_gradient_flow(self, model: BrainMAE) -> Dict:
        """Check if gradients flow through all layers"""
        results = {
            "passed": True,
            "checks": {},
            "gradient_norms": {}
        }
        
        model.train()
        batch_size = 2
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        
        x = torch.randn(batch_size, n_patches, patch_dim, requires_grad=True).to(self.device)
        
        try:
            output = model(x)
            loss = model.compute_loss(output["reconstruction"], x, output["mask"])
            loss.backward()
            
            # Check gradients for each layer
            dead_layers = []
            exploding_layers = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    results["gradient_norms"][name] = grad_norm
                    
                    if grad_norm < 1e-8:
                        dead_layers.append(name)
                    elif grad_norm > 1000:
                        exploding_layers.append(name)
            
            results["checks"]["no_dead_layers"] = len(dead_layers) == 0
            results["checks"]["no_exploding_layers"] = len(exploding_layers) == 0
            results["dead_layers"] = dead_layers
            results["exploding_layers"] = exploding_layers
            
            # Summary stats
            grad_values = list(results["gradient_norms"].values())
            results["grad_stats"] = {
                "mean": float(np.mean(grad_values)),
                "max": float(np.max(grad_values)),
                "min": float(np.min(grad_values)),
                "std": float(np.std(grad_values))
            }
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_single_batch_overfit(
        self, 
        model: BrainMAE, 
        n_steps: int = 100,
        target_loss_ratio: float = 0.5  # 50% reduction is good enough for sanity check
    ) -> Dict:
        """
        Test if model can overfit a single batch.
        This is the #1 sanity check - if this fails, something is fundamentally wrong.
        """
        results = {
            "passed": True,
            "checks": {},
            "loss_history": []
        }
        
        model.train()
        batch_size = 4
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        
        # Fixed batch
        x = torch.randn(batch_size, n_patches, patch_dim).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = None
        
        try:
            for step in range(n_steps):
                optimizer.zero_grad()
                output = model(x)
                loss = model.compute_loss(output["reconstruction"], x, output["mask"])
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                results["loss_history"].append(loss_val)
                
                if step == 0:
                    initial_loss = loss_val
            
            final_loss = results["loss_history"][-1]
            
            # Check if loss decreased significantly
            loss_ratio = final_loss / (initial_loss + 1e-8)
            results["checks"]["loss_decreased"] = final_loss < initial_loss
            results["checks"]["significant_decrease"] = loss_ratio < target_loss_ratio
            
            results["initial_loss"] = initial_loss
            results["final_loss"] = final_loss
            results["loss_ratio"] = loss_ratio
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_masking_behavior(self, model: BrainMAE) -> Dict:
        """Verify masking works correctly"""
        results = {
            "passed": True,
            "checks": {}
        }
        
        model.eval()
        batch_size = 4
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        mask_ratio = self.config.model.mask_ratio
        
        x = torch.randn(batch_size, n_patches, patch_dim).to(self.device)
        
        try:
            with torch.no_grad():
                output = model(x)
            
            mask = output["mask"]
            
            # Check mask ratio
            actual_mask_ratio = mask.float().mean().item()
            results["checks"]["correct_mask_ratio"] = abs(actual_mask_ratio - mask_ratio) < 0.05
            results["actual_mask_ratio"] = actual_mask_ratio
            results["expected_mask_ratio"] = mask_ratio
            
            # Check mask is binary
            unique_values = mask.unique()
            results["checks"]["binary_mask"] = len(unique_values) == 2
            
            # Check different masks per sample (randomness)
            masks_different = not torch.all(mask[0] == mask[1]).item()
            results["checks"]["random_masks"] = masks_different
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def check_reconstruction_quality(self, model: BrainMAE) -> Dict:
        """Check reconstruction without masking (should be near-perfect)"""
        results = {
            "passed": True,
            "checks": {}
        }
        
        model.eval()
        batch_size = 4
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        
        x = torch.randn(batch_size, n_patches, patch_dim).to(self.device)
        
        try:
            with torch.no_grad():
                # No masking
                output = model(x, mask_ratio=0.0)
            
            recon = output["reconstruction"]
            
            # With no masking, reconstruction should pass through encoder-decoder
            # It won't be perfect without training, but should be reasonable
            mse = ((recon - x) ** 2).mean().item()
            correlation = torch.corrcoef(
                torch.stack([x.flatten(), recon.flatten()])
            )[0, 1].item()
            
            results["mse_no_mask"] = mse
            results["correlation_no_mask"] = correlation
            
            # These are loose checks for untrained model
            results["checks"]["finite_mse"] = np.isfinite(mse)
            results["checks"]["reasonable_output"] = mse < 100  # Very loose
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            return results
        
        results["passed"] = all(results["checks"].values())
        return results
    
    def run_learning_rate_finder(
        self, 
        model: BrainMAE,
        init_lr: float = 1e-7,
        final_lr: float = 1.0,
        n_steps: int = 100
    ) -> Dict:
        """
        Find optimal learning rate using LR range test.
        Expert technique: Always run this before training!
        """
        results = {
            "lrs": [],
            "losses": [],
            "suggested_lr": None
        }
        
        model.train()
        batch_size = 4
        n_patches = self.config.patch.n_patches
        patch_dim = self.config.patch.patch_dim
        
        # Generate batches
        batches = [
            torch.randn(batch_size, n_patches, patch_dim).to(self.device)
            for _ in range(n_steps)
        ]
        
        lr = init_lr
        lr_mult = (final_lr / init_lr) ** (1 / n_steps)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        min_loss = float('inf')
        
        for step, x in enumerate(batches):
            optimizer.zero_grad()
            output = model(x)
            loss = model.compute_loss(output["reconstruction"], x, output["mask"])
            
            loss_val = loss.item()
            results["lrs"].append(lr)
            results["losses"].append(loss_val)
            
            if loss_val < min_loss:
                min_loss = loss_val
            
            # Stop if loss explodes
            if loss_val > 4 * min_loss or not np.isfinite(loss_val):
                break
            
            loss.backward()
            optimizer.step()
            
            # Update LR
            lr *= lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Find optimal LR (steepest descent)
        if len(results["losses"]) > 10:
            losses = np.array(results["losses"])
            lrs = np.array(results["lrs"])
            
            # Smooth losses
            smoothed = np.convolve(losses, np.ones(5)/5, mode='valid')
            
            # Find steepest descent
            gradients = np.gradient(smoothed)
            min_grad_idx = np.argmin(gradients)
            
            # Suggested LR is slightly before minimum
            suggested_idx = max(0, min_grad_idx - 5)
            results["suggested_lr"] = float(lrs[suggested_idx])
        
        return results
    
    def run_all_checks(self) -> Dict:
        """Run all model validation checks"""
        print("\n" + "=" * 70)
        print("MODEL VALIDATION")
        print("=" * 70)
        
        # Create fresh model for testing
        model = create_model(self.config).to(self.device)
        
        all_results = {
            "passed": True,
            "parameter_count": count_parameters(model)
        }
        
        # 1. Forward pass
        print("\n1. Forward pass check...")
        forward_result = self.check_forward_pass(model)
        all_results["forward_pass"] = forward_result
        status = "✓" if forward_result["passed"] else "✗"
        print(f"   {status} Forward pass: {forward_result['checks']}")
        if not forward_result["passed"]:
            all_results["passed"] = False
        
        # 2. Gradient flow
        print("\n2. Gradient flow check...")
        # Reset model
        model = create_model(self.config).to(self.device)
        gradient_result = self.check_gradient_flow(model)
        all_results["gradient_flow"] = {
            "passed": gradient_result["passed"],
            "checks": gradient_result["checks"],
            "grad_stats": gradient_result.get("grad_stats", {}),
            "dead_layers": gradient_result.get("dead_layers", []),
            "exploding_layers": gradient_result.get("exploding_layers", [])
        }
        status = "✓" if gradient_result["passed"] else "✗"
        print(f"   {status} Gradient flow: {gradient_result['checks']}")
        if gradient_result.get("grad_stats"):
            print(f"      Mean grad norm: {gradient_result['grad_stats']['mean']:.6f}")
        if not gradient_result["passed"]:
            all_results["passed"] = False
        
        # 3. Single batch overfit
        print("\n3. Single batch overfit test...")
        model = create_model(self.config).to(self.device)
        overfit_result = self.check_single_batch_overfit(model)
        all_results["single_batch_overfit"] = {
            "passed": overfit_result["passed"],
            "checks": overfit_result["checks"],
            "initial_loss": overfit_result.get("initial_loss"),
            "final_loss": overfit_result.get("final_loss"),
            "loss_ratio": overfit_result.get("loss_ratio")
        }
        status = "✓" if overfit_result["passed"] else "✗"
        print(f"   {status} Overfit test: {overfit_result['checks']}")
        if overfit_result.get("initial_loss"):
            print(f"      Loss: {overfit_result['initial_loss']:.4f} → {overfit_result['final_loss']:.4f}")
        if not overfit_result["passed"]:
            all_results["passed"] = False
            print("   ⚠️  Model cannot learn! Check architecture.")
        
        # 4. Masking behavior
        print("\n4. Masking behavior check...")
        model = create_model(self.config).to(self.device)
        mask_result = self.check_masking_behavior(model)
        all_results["masking"] = mask_result
        status = "✓" if mask_result["passed"] else "✗"
        print(f"   {status} Masking: {mask_result['checks']}")
        print(f"      Actual mask ratio: {mask_result.get('actual_mask_ratio', 'N/A'):.2%}")
        if not mask_result["passed"]:
            all_results["passed"] = False
        
        # 5. Reconstruction quality
        print("\n5. Reconstruction quality check...")
        recon_result = self.check_reconstruction_quality(model)
        all_results["reconstruction"] = recon_result
        status = "✓" if recon_result["passed"] else "✗"
        print(f"   {status} Reconstruction: {recon_result['checks']}")
        if not recon_result["passed"]:
            all_results["passed"] = False
        
        # Summary
        print("\n" + "=" * 70)
        if all_results["passed"]:
            print("✅ ALL MODEL VALIDATION CHECKS PASSED")
        else:
            print("❌ SOME MODEL VALIDATION CHECKS FAILED")
        print(f"   Total parameters: {all_results['parameter_count']:,}")
        print("=" * 70)
        
        return all_results


def validate_model(config: Config = None) -> Dict:
    """Main model validation entry point"""
    if config is None:
        config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validator = ModelValidator(config, device)
    return validator.run_all_checks()


if __name__ == "__main__":
    results = validate_model()
