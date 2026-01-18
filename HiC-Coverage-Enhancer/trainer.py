"""
Hi-C Enhancement Trainer
========================

Training logic for Hi-C enhancement model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import Config, create_experiment_dir
from model import HiCEnhancementModel
from dataset import HiCDataset, create_data_loaders, split_dataset
from metrics import calculate_metrics, MetricsTracker
from visualize import plot_training_curves, plot_predictions, plot_epoch_summary


class Trainer:
    """Trainer class for Hi-C enhancement model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()
        
        # Create experiment directory
        self.save_dir = create_experiment_dir(config.output_dir)
        config.save(f'{self.save_dir}/config.json')
        
        # Initialize model
        self.model = HiCEnhancementModel(
            base_channels=config.model.base_channels,
            num_transformer_layers=config.model.num_transformer_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        ).to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Training history
        self.history = {
            'train': {k: [] for k in ['loss', 'mse', 'mae', 'psnr', 'ssim', 'pearson', 'spearman']},
            'val': {k: [] for k in ['loss', 'mse', 'mae', 'psnr', 'ssim', 'pearson', 'spearman']}
        }
        
        self.best_val_ssim = 0
        self.start_epoch = 1
        
        print(f"\n{'='*70}")
        print(f"HI-C ENHANCEMENT TRAINING")
        print(f"{'='*70}")
        print(f"Output directory: {self.save_dir}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*70}\n")
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined L1 + L2 loss"""
        loss_l1 = F.l1_loss(pred, target)
        loss_l2 = F.mse_loss(pred, target)
        return self.config.training.l1_weight * loss_l1 + self.config.training.l2_weight * loss_l2
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        tracker = MetricsTracker()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (low_res, high_res) in enumerate(pbar):
            low_res, high_res = low_res.to(self.device), high_res.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                pred = self.model(low_res)
                loss = self.compute_loss(pred, high_res)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Calculate metrics periodically
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    metrics = calculate_metrics(pred, high_res)
                    tracker.update(metrics)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ssim': f'{tracker.get_latest().get("ssim", 0):.4f}'
            })
        
        avg_metrics = tracker.get_averages()
        avg_metrics['loss'] = total_loss / len(train_loader)
        
        return avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        total_loss = 0
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for low_res, high_res in tqdm(val_loader, desc="Validation"):
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                pred = self.model(low_res)
                loss = self.compute_loss(pred, high_res)
                
                total_loss += loss.item()
                
                metrics = calculate_metrics(pred, high_res)
                tracker.update(metrics)
        
        avg_metrics = tracker.get_averages()
        avg_metrics['loss'] = total_loss / len(val_loader)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_ssim': self.best_val_ssim,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            }
        }
        
        if filename:
            path = f'{self.save_dir}/{filename}'
        elif is_best:
            path = f'{self.save_dir}/best_model.pth'
        else:
            path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'best_val_ssim' in checkpoint:
            self.best_val_ssim = checkpoint['best_val_ssim']
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = train_metrics.copy()
            
            # Scheduler step
            self.scheduler.step()
            
            # Log metrics
            for k in self.history['train'].keys():
                self.history['train'][k].append(train_metrics.get(k, 0))
                self.history['val'][k].append(val_metrics.get(k, 0))
            
            # Print metrics
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, SSIM: {train_metrics['ssim']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}, Pearson: {train_metrics['pearson']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, SSIM: {val_metrics['ssim']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f}, Pearson: {val_metrics['pearson']:.4f}")
            
            # Save best model
            if val_metrics['ssim'] > self.best_val_ssim:
                self.best_val_ssim = val_metrics['ssim']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  -> Saved best model (SSIM: {self.best_val_ssim:.4f})")
            
            # Save periodic checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch)
                print(f"  -> Saved checkpoint at epoch {epoch}")
            
            # Plot curves
            if epoch % self.config.training.plot_every == 0:
                plot_training_curves(self.history, self.save_dir)
            
            # Plot predictions
            if epoch % 5 == 0 and val_loader:
                plot_predictions(self.model, val_loader, self.device, self.save_dir)
        
        # Final saves
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest validation SSIM: {self.best_val_ssim:.4f}")
        print(f"Results saved to: {self.save_dir}")
        
        self.save_checkpoint(self.config.training.num_epochs, filename='final_model.pth')
        plot_training_curves(self.history, self.save_dir)
        
        if val_loader:
            plot_predictions(self.model, val_loader, self.device, self.save_dir, num_samples=10)
        
        # Save history
        with open(f'{self.save_dir}/history.json', 'w') as f:
            json.dump({k: {kk: [float(x) for x in vv] 
                          for kk, vv in v.items()} 
                      for k, v in self.history.items()}, f, indent=2)
        
        print("\nTraining complete! Model ready for enhancement.")
        return self.history


def train_model(
    train_data: str,
    val_data: Optional[str] = None,
    config: Optional[Config] = None,
    resume_from: Optional[str] = None
) -> str:
    """
    Convenience function to train a model
    
    Args:
        train_data: Path to training data .npz file
        val_data: Path to validation data .npz file (optional)
        config: Configuration object (uses defaults if None)
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Path to experiment directory
    """
    if config is None:
        config = Config()
    
    config.data.train_data = train_data
    if val_data:
        config.data.val_data = val_data
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_path=config.data.train_data,
        val_path=config.data.val_data if val_data else None,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        augment=config.training.augment
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return trainer.save_dir
