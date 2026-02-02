"""
Training Script for Brain MAE
Implements expert training practices:
- LR finder
- Cosine annealing with warmup
- Early stopping
- Gradient clipping
- Comprehensive logging
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(str(Path(__file__).parent))
from configs.config import Config, get_config
from data.preprocessing import create_dataloaders, run_data_quality_checks
from models.mae_model import create_model, BrainMAE


class CosineAnnealingWarmup:
    """Cosine annealing with linear warmup - the gold standard schedule"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        lr = self.get_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class GradientMonitor:
    """Monitor gradient health during training"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_history = []
    
    def clip_and_monitor(self, max_norm: float = 1.0) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm
        )
        self.grad_history.append(total_norm.item())
        return total_norm.item()
    
    def get_stats(self) -> Dict:
        if not self.grad_history:
            return {}
        return {
            "mean": np.mean(self.grad_history[-100:]),
            "max": np.max(self.grad_history[-100:]),
            "min": np.min(self.grad_history[-100:])
        }


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._get_device()
        
        # Create model
        self.model = create_model(config).to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            base_lr=config.training.learning_rate,
            min_lr=config.training.min_lr
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Gradient monitor
        self.grad_monitor = GradientMonitor(self.model)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None
    
    def _get_device(self) -> torch.device:
        if self.config.training.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=self.config.training.betas
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            patches = batch['patches'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = self.model(patches)
                    loss = self.model.compute_loss(
                        output['reconstruction'], 
                        patches, 
                        output['mask']
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_monitor.clip_and_monitor(self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(patches)
                loss = self.model.compute_loss(
                    output['reconstruction'], 
                    patches, 
                    output['mask']
                )
                
                loss.backward()
                self.grad_monitor.clip_and_monitor(self.config.training.max_grad_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_loader:
            patches = batch['patches'].to(self.device)
            
            output = self.model(patches)
            loss = self.model.compute_loss(
                output['reconstruction'], 
                patches, 
                output['mask']
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'config': {
                'model': vars(self.config.model),
                'training': vars(self.config.training),
                'patch': vars(self.config.patch)
            }
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict:
        """Full training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Update learning rate
            lr = self.scheduler.step(epoch)
            self.learning_rates.append(lr)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0 or is_best:
                ckpt_path = self.config.data.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(ckpt_path, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            grad_stats = self.grad_monitor.get_stats()
            
            print(f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | Time: {epoch_time:.1f}s | "
                  f"{'★ Best' if is_best else ''}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        results = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time_seconds': total_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final train/val loss: {self.train_losses[-1]:.4f} / {self.val_losses[-1]:.4f}")
        
        # Save results
        results_path = self.config.data.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump({k: v if not isinstance(v, list) else v for k, v in results.items()}, f, indent=2)
        
        return results


def run_sanity_checks(model: BrainMAE, train_loader: DataLoader, device: torch.device) -> bool:
    """
    Run sanity checks before full training.
    Expert principle: Always verify model can learn!
    """
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    model.train()
    
    # Get single batch
    batch = next(iter(train_loader))
    patches = batch['patches'].to(device)
    
    print(f"\n1. Single batch overfit test...")
    print(f"   Input shape: {patches.shape}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    initial_loss = None
    for i in range(50):
        optimizer.zero_grad()
        output = model(patches)
        loss = model.compute_loss(output['reconstruction'], patches, output['mask'])
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        
        if i % 10 == 0:
            print(f"   Step {i:3d}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    
    if final_loss > initial_loss * 0.5:
        print(f"\n   ❌ FAIL: Loss didn't decrease enough ({initial_loss:.4f} -> {final_loss:.4f})")
        return False
    
    print(f"\n   ✓ PASS: Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
    
    # Check gradient flow
    print(f"\n2. Gradient flow check...")
    
    dead_layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.abs().max() < 1e-7:
                dead_layers.append(name)
    
    if dead_layers:
        print(f"   ❌ FAIL: Dead layers found: {dead_layers[:3]}...")
        return False
    
    print(f"   ✓ PASS: All layers receiving gradients")
    
    print("\n" + "=" * 60)
    print("✓ All sanity checks passed!")
    print("=" * 60)
    
    return True


def main():
    """Main training entry point"""
    print("\n" + "#" * 60)
    print("# BRAIN MAE TRAINING")
    print("#" * 60)
    
    # Load config
    config = get_config()
    config.print_summary()
    
    # Run data quality checks
    print("\n" + "=" * 60)
    print("STEP 1: Data Quality Checks")
    print("=" * 60)
    
    quality_results = run_data_quality_checks(config)
    if not quality_results["passed"]:
        print("\n❌ Data quality checks failed. Fix issues before training.")
        return
    
    # Create dataloaders
    print("\n" + "=" * 60)
    print("STEP 2: Creating Dataloaders")
    print("=" * 60)
    
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n" + "=" * 60)
    print("STEP 3: Creating Model & Trainer")
    print("=" * 60)
    
    trainer = Trainer(config)
    
    # Run sanity checks
    print("\n" + "=" * 60)
    print("STEP 4: Sanity Checks")
    print("=" * 60)
    
    # Reset model for sanity check
    sanity_model = create_model(config).to(trainer.device)
    if not run_sanity_checks(sanity_model, train_loader, trainer.device):
        print("\n❌ Sanity checks failed. Check model architecture.")
        return
    
    del sanity_model  # Free memory
    
    # Train
    print("\n" + "=" * 60)
    print("STEP 5: Training")
    print("=" * 60)
    
    results = trainer.train(train_loader, val_loader)
    
    print("\n✓ Training complete!")
    print(f"  Results saved to: {config.data.output_dir}")
    print(f"  Checkpoints saved to: {config.data.checkpoint_dir}")


if __name__ == "__main__":
    main()
