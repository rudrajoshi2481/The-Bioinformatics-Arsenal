"""
Hi-C Enhancement Visualization
==============================

Visualization utilities for training and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import torch
from skimage.metrics import structural_similarity as ssim


def plot_training_curves(history: Dict, save_dir: str, filename: str = 'training_curves.png'):
    """
    Plot comprehensive training curves
    
    Args:
        history: Dictionary with 'train' and 'val' keys, each containing metric lists
        save_dir: Directory to save the plot
        filename: Output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    metrics = ['loss', 'ssim', 'psnr', 'pearson', 'mae', 'mse']
    titles = ['Loss', 'SSIM', 'PSNR (dB)', 'Pearson Correlation', 'MAE', 'MSE']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        if metric in history['train'] and len(history['train'][metric]) > 0:
            epochs = range(1, len(history['train'][metric]) + 1)
            ax.plot(epochs, history['train'][metric], 'b-', label='Train', linewidth=2)
            
            if 'val' in history and metric in history['val'] and len(history['val'][metric]) > 0:
                ax.plot(epochs, history['val'][metric], 'r-', label='Validation', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 5,
    filename: str = 'predictions.png'
):
    """
    Visualize prediction examples
    
    Args:
        model: Trained model
        data_loader: Data loader with samples
        device: Torch device
        save_dir: Directory to save the plot
        num_samples: Number of samples to visualize
        filename: Output filename
    """
    model.eval()
    
    # Get samples
    low_res, high_res = next(iter(data_loader))
    low_res = low_res[:num_samples].to(device)
    high_res = high_res[:num_samples].to(device)
    
    with torch.no_grad():
        pred = model(low_res)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    fig.suptitle('Enhancement Results', fontsize=16, fontweight='bold')
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Low-res input
        axes[i, 0].imshow(low_res[i, 0].cpu(), cmap='Reds', vmin=0, vmax=1)
        axes[i, 0].set_title('Low-Res Input', fontsize=12)
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(pred[i, 0].cpu(), cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title('Enhanced (Ours)', fontsize=12)
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(high_res[i, 0].cpu(), cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth', fontsize=12)
        axes[i, 2].axis('off')
        
        # Difference
        diff = torch.abs(pred[i, 0] - high_res[i, 0]).cpu()
        im = axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 3].set_title('Absolute Error', fontsize=12)
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        
        # Calculate metrics
        try:
            ssim_score = ssim(
                high_res[i, 0].cpu().numpy(),
                pred[i, 0].cpu().numpy(),
                data_range=1.0
            )
            axes[i, 1].text(0.5, -0.1, f'SSIM: {ssim_score:.4f}',
                           transform=axes[i, 1].transAxes,
                           ha='center', fontsize=10, fontweight='bold')
        except Exception:
            pass
    
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hic_matrix(
    matrix: np.ndarray,
    save_path: str,
    title: str = 'Hi-C Contact Matrix',
    cmap: str = 'Reds',
    vmax: Optional[float] = None
):
    """
    Plot a single Hi-C contact matrix
    
    Args:
        matrix: 2D numpy array
        save_path: Path to save the figure
        title: Plot title
        cmap: Colormap
        vmax: Maximum value for colormap
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if vmax is None:
        vmax = np.percentile(matrix, 98)
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(
    low_res: np.ndarray,
    enhanced: np.ndarray,
    high_res: np.ndarray,
    save_path: str,
    title: str = 'Hi-C Enhancement Comparison'
):
    """
    Plot comparison of low-res, enhanced, and high-res matrices
    
    Args:
        low_res: Low resolution input matrix
        enhanced: Enhanced output matrix
        high_res: High resolution target matrix
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    vmax = np.percentile(high_res, 98)
    
    # Low-res
    im0 = axes[0].imshow(low_res, cmap='Reds', vmin=0, vmax=vmax)
    axes[0].set_title('Low Coverage', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Enhanced
    im1 = axes[1].imshow(enhanced, cmap='Reds', vmin=0, vmax=vmax)
    axes[1].set_title('Enhanced', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # High-res
    im2 = axes[2].imshow(high_res, cmap='Reds', vmin=0, vmax=vmax)
    axes[2].set_title('High Coverage (Target)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Difference
    diff = np.abs(enhanced - high_res)
    im3 = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=vmax/2)
    axes[3].set_title('Absolute Error', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_epoch_summary(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    save_dir: str
):
    """
    Plot summary for a single epoch
    
    Args:
        epoch: Current epoch number
        train_metrics: Training metrics
        val_metrics: Validation metrics
        save_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Epoch {epoch} Summary', fontsize=14, fontweight='bold')
    
    metrics = ['loss', 'ssim', 'psnr', 'pearson']
    titles = ['Loss', 'SSIM', 'PSNR', 'Pearson']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        train_val = train_metrics.get(metric, 0)
        val_val = val_metrics.get(metric, 0)
        
        bars = ax.bar(['Train', 'Val'], [train_val, val_val], color=['blue', 'red'])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title)
        
        for bar, val in zip(bars, [train_val, val_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/epoch_{epoch:03d}_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
