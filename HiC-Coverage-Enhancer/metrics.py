"""
Hi-C Enhancement Metrics
========================

Metrics calculation for Hi-C enhancement evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for Hi-C enhancement
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
    
    Returns:
        Dictionary with metric names and values
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    metrics = {}
    
    # MSE
    metrics['mse'] = float(F.mse_loss(pred, target).item())
    
    # MAE
    metrics['mae'] = float(F.l1_loss(pred, target).item())
    
    # PSNR
    mse = metrics['mse']
    if mse > 0:
        metrics['psnr'] = 10 * np.log10(1.0 / mse)
    else:
        metrics['psnr'] = 100.0
    
    # SSIM (batch average)
    ssim_scores = []
    for i in range(pred_np.shape[0]):
        try:
            score = ssim(
                target_np[i, 0],
                pred_np[i, 0],
                data_range=1.0
            )
            ssim_scores.append(score)
        except Exception:
            continue
    metrics['ssim'] = np.mean(ssim_scores) if ssim_scores else 0.0
    
    # Pearson correlation
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    try:
        metrics['pearson'], _ = pearsonr(pred_flat, target_flat)
    except Exception:
        metrics['pearson'] = 0.0
    
    # Spearman correlation
    try:
        metrics['spearman'], _ = spearmanr(pred_flat, target_flat)
    except Exception:
        metrics['spearman'] = 0.0
    
    return metrics


def calculate_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate SSIM for single matrices"""
    return ssim(target, pred, data_range=1.0)


def calculate_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate PSNR for single matrices"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.metrics_keys = ['loss', 'mse', 'mae', 'psnr', 'ssim', 'pearson', 'spearman']
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.values = {k: [] for k in self.metrics_keys}
    
    def update(self, metrics: Dict[str, float]):
        """Add new metrics values"""
        for k, v in metrics.items():
            if k in self.values:
                self.values[k].append(v)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average of all tracked metrics"""
        return {k: np.mean(v) if v else 0.0 for k, v in self.values.items()}
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values"""
        return {k: v[-1] if v else 0.0 for k, v in self.values.items()}
