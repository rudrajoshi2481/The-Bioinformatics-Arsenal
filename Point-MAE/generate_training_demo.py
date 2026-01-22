#!/usr/bin/env python3
"""
Generate training demonstration visualization.

Creates a grid showing airplane reconstruction at different training epochs
to demonstrate the model learning to reconstruct masked point clouds.

Usage:
    python generate_training_demo.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

from models import PointMAE, generate_random_object


def train_and_visualize(output_dir='images', epochs_to_show=[1, 5, 10, 25, 50, 100]):
    """Train model and save visualizations at specific epochs."""
    
    device = torch.device('cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Point-MAE Training Demo - Airplane Reconstruction")
    print("=" * 60)
    
    # Generate airplane point cloud
    airplane_pts = generate_random_object(num_points=1024, obj_type='airplane', with_normals=True)
    airplane_tensor = torch.from_numpy(airplane_pts).unsqueeze(0).float().to(device)
    
    print(f"Airplane points: {airplane_pts.shape}")
    
    # Create model
    model = PointMAE(
        num_group=32,
        group_size=16,
        trans_dim=256,
        depth=4,
        decoder_depth=2,
        num_heads=4,
        decoder_num_heads=4,
        encoder_dims=256,
        input_channel=6,
        mask_ratio=0.6,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # Store visualizations
    vis_data = {}
    max_epoch = max(epochs_to_show)
    
    print(f"\nTraining for {max_epoch} epochs...")
    print(f"Saving visualizations at epochs: {epochs_to_show}")
    
    # Save initial (untrained) visualization
    model.eval()
    with torch.no_grad():
        full_rec, vis_pts, _ = model(airplane_tensor, vis=True)
    vis_data[0] = {
        'input': airplane_pts[:, :3],
        'visible': vis_pts[0].cpu().numpy(),
        'reconstructed': full_rec[0].cpu().numpy(),
        'loss': float('inf')
    }
    
    # Training loop
    losses = []
    pbar = tqdm(range(1, max_epoch + 1), desc="Training")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        loss = model(airplane_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Save visualization at specific epochs
        if epoch in epochs_to_show:
            model.eval()
            with torch.no_grad():
                full_rec, vis_pts, _ = model(airplane_tensor, vis=True)
            vis_data[epoch] = {
                'input': airplane_pts[:, :3],
                'visible': vis_pts[0].cpu().numpy(),
                'reconstructed': full_rec[0].cpu().numpy(),
                'loss': loss.item()
            }
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    
    # Create visualization grid
    print("\nGenerating visualization grid...")
    
    epochs_with_zero = [0] + epochs_to_show
    n_cols = len(epochs_with_zero)
    
    fig = plt.figure(figsize=(4 * n_cols, 12))
    
    for col, epoch in enumerate(epochs_with_zero):
        data = vis_data[epoch]
        
        # Row 1: Input
        ax1 = fig.add_subplot(3, n_cols, col + 1, projection='3d')
        pts = data['input']
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, alpha=0.6)
        ax1.set_title(f'Epoch {epoch}\nInput', fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        
        # Row 2: Visible (40% unmasked)
        ax2 = fig.add_subplot(3, n_cols, n_cols + col + 1, projection='3d')
        pts = data['visible']
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='green', s=2, alpha=0.7)
        ax2.set_title(f'Visible (40%)', fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        
        # Row 3: Reconstructed
        ax3 = fig.add_subplot(3, n_cols, 2 * n_cols + col + 1, projection='3d')
        pts = data['reconstructed']
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=2, alpha=0.7)
        loss_str = f"Loss: {data['loss']:.4f}" if data['loss'] != float('inf') else "Untrained"
        ax3.set_title(f'Reconstructed\n{loss_str}', fontsize=10)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
    
    plt.suptitle('Point-MAE Training Progress: Airplane Reconstruction\n(60% masked, 40% visible)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'training_progress.png'}")
    
    # Create loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1)
    ax.scatter(epochs_to_show, [losses[e-1] for e in epochs_to_show], c='red', s=50, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chamfer Distance Loss')
    ax.set_title('Training Loss Curve')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'loss_curve.png'}")
    
    # Create a compact single-row visualization for README
    fig = plt.figure(figsize=(16, 4))
    
    key_epochs = [0, 1, 10, 50, 100]
    for i, epoch in enumerate(key_epochs):
        if epoch in vis_data:
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            pts = vis_data[epoch]['reconstructed']
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=2, alpha=0.7)
            loss = vis_data[epoch]['loss']
            loss_str = f"Loss: {loss:.4f}" if loss != float('inf') else "Untrained"
            ax.set_title(f'Epoch {epoch}\n{loss_str}', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=20, azim=45)
    
    plt.suptitle('Point-MAE: Learning to Reconstruct Masked Point Clouds', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'reconstruction_progress.png'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return output_dir


if __name__ == '__main__':
    train_and_visualize()
