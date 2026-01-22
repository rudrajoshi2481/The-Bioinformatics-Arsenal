#!/usr/bin/env python3
"""
Point-MAE Training Script

A simple, CPU-compatible training script for Point-MAE with:
- Comprehensive logging
- Visualization of reconstructions
- Random object generation for testing
- Validation loop

Usage:
    python train.py --epochs 50 --batch_size 4
    python train.py --use_random_objects --epochs 100
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import argparse
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import PointMAE, generate_random_object


class RandomObjectDataset(Dataset):
    """Dataset that generates random 3D objects on the fly."""
    
    def __init__(self, num_samples=1000, num_points=1024, with_normals=True):
        self.num_samples = num_samples
        self.num_points = num_points
        self.with_normals = with_normals
        self.object_types = ['sphere', 'cube', 'cylinder', 'torus', 'airplane', 'chair']
        
        # Pre-generate for consistency
        self.data = []
        for i in range(num_samples):
            obj_type = self.object_types[i % len(self.object_types)]
            pts = generate_random_object(num_points, obj_type, with_normals)
            self.data.append((pts, i % len(self.object_types)))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        pts, label = self.data[idx]
        return torch.from_numpy(pts).float(), label


class PointCloudDataset(Dataset):
    """Dataset for loading point clouds from .dat files."""
    
    def __init__(self, data_path, split='train', num_points=1024, with_normals=True):
        import pickle
        
        self.num_points = num_points
        self.with_normals = with_normals
        
        # Load data
        dat_file = os.path.join(data_path, f'modelnet40_{split}_8192pts_fps.dat')
        
        if os.path.exists(dat_file):
            print(f"Loading {split} data from {dat_file}...")
            with open(dat_file, 'rb') as f:
                raw_data = pickle.load(f)
            self.data = self._parse_data(raw_data)
            print(f"Loaded {len(self.data)} samples")
        else:
            raise FileNotFoundError(f"Dataset not found: {dat_file}")
    
    def _parse_data(self, raw_data):
        """Parse raw data into list of (points, label) tuples."""
        parsed = []
        
        def is_valid(pts):
            if pts is None or not hasattr(pts, 'shape'):
                return False
            if pts.ndim != 2 or pts.shape[0] < 100:
                return False
            if pts.shape[1] not in [3, 6]:
                return False
            return True
        
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first = raw_data[0]
            
            if hasattr(first, 'shape') and first.ndim == 2:
                for i, pts in enumerate(raw_data):
                    if is_valid(pts):
                        parsed.append((pts, i % 40))
            elif isinstance(first, (list, tuple)):
                for item in raw_data:
                    if isinstance(item, (list, tuple)):
                        for pts in item:
                            pts = np.array(pts)
                            if is_valid(pts):
                                parsed.append((pts, len(parsed) % 40))
        
        return parsed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pts, label = self.data[idx]
        pts = pts.copy()
        
        # Sample points
        if pts.shape[0] > self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts = pts[choice]
        elif pts.shape[0] < self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
            pts = pts[choice]
        
        # Normalize
        centroid = pts[:, :3].mean(axis=0)
        pts[:, :3] = pts[:, :3] - centroid
        max_dist = np.max(np.linalg.norm(pts[:, :3], axis=1))
        if max_dist > 0:
            pts[:, :3] = pts[:, :3] / max_dist
        
        # Select channels
        if not self.with_normals and pts.shape[1] > 3:
            pts = pts[:, :3]
        
        return torch.from_numpy(pts).float(), label


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV file
        self.csv_path = self.log_dir / f'metrics_{self.timestamp}.csv'
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'time'])
        
        # Text log
        self.log_path = self.log_dir / f'train_{self.timestamp}.log'
        
    def log(self, msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
    
    def log_metrics(self, epoch, step, train_loss, val_loss=None, lr=None):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, train_loss, val_loss, lr, datetime.now().isoformat()])


def visualize_reconstruction(model, pts, save_path, device='cpu'):
    """Visualize model reconstruction."""
    model.eval()
    
    with torch.no_grad():
        pts_tensor = pts.unsqueeze(0).to(device) if pts.dim() == 2 else pts.to(device)
        full_rec, vis_pts, centers = model(pts_tensor, vis=True)
    
    # Convert to numpy
    input_xyz = pts_tensor[0, :, :3].cpu().numpy()
    rec_xyz = full_rec[0].cpu().numpy()
    vis_xyz = vis_pts[0].cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Input
    ax1 = fig.add_subplot(131, projection='3d')
    idx = np.random.choice(len(input_xyz), min(1024, len(input_xyz)), replace=False)
    ax1.scatter(input_xyz[idx, 0], input_xyz[idx, 1], input_xyz[idx, 2], 
                c='blue', s=1, alpha=0.6)
    ax1.set_title(f'Input ({len(input_xyz)} pts)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Visible
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(vis_xyz[:, 0], vis_xyz[:, 1], vis_xyz[:, 2], 
                c='green', s=2, alpha=0.7)
    ax2.set_title(f'Visible ({len(vis_xyz)} pts)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Reconstruction
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(rec_xyz[:, 0], rec_xyz[:, 1], rec_xyz[:, 2], 
                c='red', s=2, alpha=0.7)
    ax3.set_title(f'Reconstructed ({len(rec_xyz)} pts)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return input_xyz, vis_xyz, rec_xyz


def plot_training_curves(train_losses, val_losses, lr_history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(train_losses, 'b-', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss
    if val_losses:
        val_steps, val_vals = zip(*val_losses)
        axes[1].plot(val_steps, val_vals, 'r-o', markersize=4)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(lr_history, 'g-')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('LR')
    axes[2].set_title('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Logger
    logger = TrainingLogger(output_dir / 'logs')
    
    logger.log("=" * 60)
    logger.log("Point-MAE Training")
    logger.log("=" * 60)
    logger.log(f"Device: {device}")
    logger.log(f"Output: {output_dir}")
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    logger.log("\nCreating datasets...")
    
    if args.use_random_objects:
        train_dataset = RandomObjectDataset(
            num_samples=args.num_train_samples,
            num_points=args.num_points,
            with_normals=True
        )
        val_dataset = RandomObjectDataset(
            num_samples=args.num_val_samples,
            num_points=args.num_points,
            with_normals=True
        )
        logger.log(f"Using random objects: {args.num_train_samples} train, {args.num_val_samples} val")
    else:
        train_dataset = PointCloudDataset(
            args.data_path, split='train', 
            num_points=args.num_points, with_normals=True
        )
        val_dataset = PointCloudDataset(
            args.data_path, split='test',
            num_points=args.num_points, with_normals=True
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.log(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.log(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Create model
    logger.log("\nCreating model...")
    model = PointMAE(
        num_group=args.num_group,
        group_size=args.group_size,
        trans_dim=args.trans_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        decoder_num_heads=args.decoder_num_heads,
        encoder_dims=args.trans_dim,
        input_channel=6,
        mask_ratio=args.mask_ratio,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.min_lr)
    
    # Training state
    global_step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    lr_history = []
    
    logger.log("\n" + "=" * 60)
    logger.log("Starting Training")
    logger.log("=" * 60)
    logger.log(f"Epochs: {args.epochs}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.lr}")
    logger.log(f"Mask ratio: {args.mask_ratio}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (pts, _) in enumerate(pbar):
            pts = pts.to(device)
            
            optimizer.zero_grad()
            loss = model(pts)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            lr_history.append(scheduler.get_last_lr()[0])
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log metrics
            if global_step % args.log_every == 0:
                logger.log_metrics(epoch, global_step, loss.item(), lr=scheduler.get_last_lr()[0])
        
        epoch_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start
        
        logger.log(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} - Time: {epoch_time:.1f}s")
        
        # Validation
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for pts, _ in val_loader:
                    pts = pts.to(device)
                    loss = model(pts)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append((global_step, val_loss))
            
            logger.log(f"  Val Loss: {val_loss:.4f}")
            logger.log_metrics(epoch, global_step, epoch_loss, val_loss, scheduler.get_last_lr()[0])
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, output_dir / 'checkpoints' / 'best_model.pt')
                logger.log(f"  New best model saved!")
        
        # Visualization
        if (epoch + 1) % args.vis_every == 0:
            sample_pts, _ = val_dataset[0]
            vis_path = output_dir / 'visualizations' / f'epoch_{epoch+1:04d}.png'
            visualize_reconstruction(model, sample_pts, vis_path, device)
            logger.log(f"  Visualization saved: {vis_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Plot training curves
        plot_training_curves(
            train_losses, val_losses, lr_history,
            output_dir / 'logs' / 'training_curves.png'
        )
    
    logger.log("\n" + "=" * 60)
    logger.log("Training Complete!")
    logger.log(f"Best validation loss: {best_val_loss:.4f}")
    logger.log("=" * 60)
    
    # Final visualization
    logger.log("\nGenerating final visualizations...")
    for i in range(min(5, len(val_dataset))):
        sample_pts, label = val_dataset[i]
        vis_path = output_dir / 'visualizations' / f'final_sample_{i}.png'
        visualize_reconstruction(model, sample_pts, vis_path, device)
    
    logger.log(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Point-MAE Training')
    
    # Data
    parser.add_argument('--data_path', type=str, default='/app/tmp/Point-MAE/data')
    parser.add_argument('--use_random_objects', action='store_true', help='Use random objects instead of dataset')
    parser.add_argument('--num_train_samples', type=int, default=500)
    parser.add_argument('--num_val_samples', type=int, default=100)
    parser.add_argument('--num_points', type=int, default=1024)
    
    # Model
    parser.add_argument('--num_group', type=int, default=32)
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--trans_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--decoder_depth', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='./output_clean')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--vis_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
