# trainer_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

from tokenizer import ESMTokenizer
from model import ESMForMaskedLM
from config import MLMConfig
from dataset import MLMDataset


class MetricsTracker:
    """Track running averages of metrics"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_average(self, key):
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])


class TrainingPlotter:
    """Generate training plots with matplotlib"""
    def __init__(self, save_dir, rank=0):
        self.save_dir = Path(save_dir)
        self.rank = rank
        self.is_main = (rank == 0)
        
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.gpu_memory = []
        self.tokens_per_second = []
        self.steps = []
        self.val_steps = []
        self.val_accuracies = []
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def update(self, step, **kwargs):
        """Update metrics"""
        if not self.is_main:
            return
        
        self.steps.append(step)
        if 'train_loss' in kwargs:
            self.train_losses.append(kwargs['train_loss'])
        if 'learning_rate' in kwargs:
            self.learning_rates.append(kwargs['learning_rate'])
        if 'grad_norm' in kwargs:
            self.grad_norms.append(kwargs['grad_norm'])
        if 'gpu_memory' in kwargs:
            self.gpu_memory.append(kwargs['gpu_memory'])
        if 'tokens_per_second' in kwargs:
            self.tokens_per_second.append(kwargs['tokens_per_second'])
    
    def update_validation(self, step, val_loss, val_accuracy):
        """Update validation metrics"""
        if not self.is_main:
            return
        
        self.val_steps.append(step)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
    
    def plot_all(self):
        """Generate all plots"""
        if not self.is_main:
            return
        
        print("📊 Generating training plots...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Loss curves
        ax1 = plt.subplot(2, 3, 1)
        if self.train_losses:
            ax1.plot(self.steps, self.train_losses, alpha=0.4, label='Train Loss')
            if len(self.train_losses) > 10:
                window = min(50, len(self.train_losses) // 10)
                smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax1.plot(self.steps[:len(smooth)], smooth, linewidth=2, label='Train (smoothed)')
        if self.val_losses:
            ax1.plot(self.val_steps, self.val_losses, 'ro-', linewidth=2, markersize=8, label='Val Loss')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation Accuracy
        ax2 = plt.subplot(2, 3, 2)
        if self.val_accuracies:
            ax2.plot(self.val_steps, self.val_accuracies, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate
        ax3 = plt.subplot(2, 3, 3)
        if self.learning_rates:
            ax3.plot(self.steps, self.learning_rates, color='green', linewidth=2)
            ax3.set_xlabel('Step', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 4. Gradient norms
        ax4 = plt.subplot(2, 3, 4)
        if self.grad_norms:
            ax4.plot(self.steps, self.grad_norms, color='orange', alpha=0.6)
            if len(self.grad_norms) > 10:
                window = min(50, len(self.grad_norms) // 10)
                smooth = np.convolve(self.grad_norms, np.ones(window)/window, mode='valid')
                ax4.plot(self.steps[:len(smooth)], smooth, color='darkred', linewidth=2, label='Smoothed')
            ax4.set_xlabel('Step', fontsize=12)
            ax4.set_ylabel('Gradient Norm', fontsize=12)
            ax4.set_title('Gradient Norms', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. GPU Memory
        ax5 = plt.subplot(2, 3, 5)
        if self.gpu_memory:
            ax5.plot(self.steps, self.gpu_memory, color='purple', linewidth=2)
            ax5.set_xlabel('Step', fontsize=12)
            ax5.set_ylabel('GPU Memory (GB)', fontsize=12)
            ax5.set_title('GPU Memory Usage (Rank 0)', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Throughput
        ax6 = plt.subplot(2, 3, 6)
        if self.tokens_per_second:
            ax6.plot(self.steps, self.tokens_per_second, color='teal', alpha=0.6)
            if len(self.tokens_per_second) > 10:
                window = min(50, len(self.tokens_per_second) // 10)
                smooth = np.convolve(self.tokens_per_second, np.ones(window)/window, mode='valid')
                ax6.plot(self.steps[:len(smooth)], smooth, color='darkblue', linewidth=2, label='Smoothed')
            ax6.set_xlabel('Step', fontsize=12)
            ax6.set_ylabel('Tokens/Second', fontsize=12)
            ax6.set_title('Training Throughput', fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'training_dashboard.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved: {plot_path}")


class ESMTrainerDDP:
    def __init__(self, config: MLMConfig, rank: int, world_size: int, resume_from: Optional[str] = None):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
        
        if self.is_main:
            print("="*60)
            print("Initializing ESM2 DDP Trainer")
            print("="*60)
            print(f"Rank: {rank}/{world_size}")
            print(f"Device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = ESMTokenizer()
        self.model = ESMForMaskedLM(config).to(self.device)
        
        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        
        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Create dataloaders with DistributedSampler
        if self.is_main:
            print("\nCreating distributed dataloaders...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Setup optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_steps / total_steps,
            anneal_strategy='cos'
        )
        
        # Setup mixed precision training
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize metrics tracker and plotter
        self.metrics_tracker = MetricsTracker(window_size=100)
        if self.is_main:
            plot_dir = Path(config.checkpoint_dir) / "plots"
            self.plotter = TrainingPlotter(plot_dir, rank=rank)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        if self.is_main:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        if self.is_main:
            print("\n✅ DDP Trainer initialized successfully\n")
    
    def _create_dataloaders(self):
        """Create distributed dataloaders"""
        # Training dataset
        train_dataset = MLMDataset(
            json_file=self.config.train_json,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            mlm_prob=self.config.mlm_probability
        )
        
        # Validation dataset
        val_dataset = MLMDataset(
            json_file=self.config.valid_json,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            mlm_prob=self.config.mlm_probability
        )
        
        # Distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        if self.is_main:
            print(f"✅ Dataloaders created")
            print(f"   Train batches per GPU: {len(train_loader):,}")
            print(f"   Val batches per GPU: {len(val_loader):,}")
            print(f"   Total train samples: {len(train_dataset):,}")
        
        return train_loader, val_loader
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(self.epoch)
        
        epoch_loss = 0
        epoch_steps = 0
        step_start_time = time.time()
        
        if self.is_main:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}",
                dynamic_ncols=True
            )
        else:
            progress_bar = self.train_loader
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Count tokens
            batch_tokens = attention_mask.sum().item()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                ).item()
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Calculate metrics
                lr = self.scheduler.get_last_lr()[0]
                step_time = time.time() - step_start_time
                tokens_per_second = batch_tokens * self.config.gradient_accumulation_steps * self.world_size / step_time
                gpu_memory = torch.cuda.max_memory_allocated(self.device) / 1e9
                
                current_loss = loss.item() * self.config.gradient_accumulation_steps
                
                # Update metrics tracker
                self.metrics_tracker.update(
                    loss=current_loss,
                    grad_norm=grad_norm,
                    tokens_per_second=tokens_per_second
                )
                
                # Update plotter (only main process)
                if self.is_main:
                    self.plotter.update(
                        step=self.global_step,
                        train_loss=current_loss,
                        learning_rate=lr,
                        grad_norm=grad_norm,
                        gpu_memory=gpu_memory,
                        tokens_per_second=tokens_per_second
                    )
                
                # Console logging
                if self.global_step % self.config.log_every == 0 and self.is_main:
                    avg_loss = self.metrics_tracker.get_average('loss')
                    avg_grad_norm = self.metrics_tracker.get_average('grad_norm')
                    avg_tps = self.metrics_tracker.get_average('tokens_per_second')
                    
                    print(f"\n[Rank {self.rank}] Step {self.global_step}")
                    print(f"  Loss: {avg_loss:.4f}")
                    print(f"  LR: {lr:.2e}")
                    print(f"  Grad Norm: {avg_grad_norm:.4f}")
                    print(f"  Throughput: {avg_tps:.0f} tok/s (all GPUs)")
                    print(f"  GPU Memory: {gpu_memory:.2f} GB")
                
                if self.is_main:
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })
                
                step_start_time = time.time()
                
                # Save checkpoint
                if self.global_step % self.config.save_every_n_steps == 0 and self.is_main:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                    self.plotter.plot_all()
        
        avg_epoch_loss = epoch_loss / epoch_steps * self.config.gradient_accumulation_steps
        return avg_epoch_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        total_correct = 0
        total_masked = 0
        
        if self.is_main:
            progress_bar = tqdm(self.val_loader, desc="Validating", dynamic_ncols=True)
        else:
            progress_bar = self.val_loader
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
            
            total_loss += loss.item()
            total_steps += 1
            
            # Calculate accuracy
            predictions = outputs['logits'].argmax(dim=-1)
            mask = labels != -100
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
            
            if self.is_main:
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        # Average across all GPUs
        avg_loss = total_loss / total_steps
        avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.world_size
        
        # Calculate accuracy
        total_correct_tensor = torch.tensor(total_correct).to(self.device)
        total_masked_tensor = torch.tensor(total_masked).to(self.device)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_masked_tensor, op=dist.ReduceOp.SUM)
        
        accuracy = (total_correct_tensor.item() / total_masked_tensor.item() * 100) if total_masked_tensor.item() > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        if self.is_main:
            print(f"\nValidation Results:")
            print(f"  Val Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Perplexity: {perplexity:.2f}")
            
            # Update plotter
            self.plotter.update_validation(self.global_step, avg_loss, accuracy)
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        if self.is_main:
            print("="*60)
            print("Starting DDP Training")
            print("="*60)
            print(f"Total epochs: {self.config.num_epochs}")
            print(f"Steps per epoch (per GPU): {len(self.train_loader)}")
            print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            print(f"Batch size per GPU: {self.config.batch_size}")
            print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size}")
            print(f"Mixed precision: {self.use_amp}")
            print(f"World size: {self.world_size}")
            print()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            if self.is_main:
                print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            if self.is_main:
                print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
            
            # Save best model (only main process)
            if self.is_main:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model')
                    print(f"✅ New best model saved (val_loss: {val_loss:.4f})")
                
                # Save epoch checkpoint
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
                
                # Generate plots
                self.plotter.plot_all()
                print()
        
        if self.is_main:
            print("="*60)
            print("Training completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print("="*60)
            
            # Final plots
            self.plotter.plot_all()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint (only main process)"""
        if not self.is_main:
            return
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if self.is_main:
            print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Map to current device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load scaler if using AMP
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.is_main:
            print(f"✅ Checkpoint loaded (epoch {self.epoch}, step {self.global_step})")


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


# main_ddp.py
def main():
    """Main training function with DDP"""
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Load config
    from config import MLMConfig
    config = MLMConfig()
    
    if rank == 0:
        print("="*60)
        print("ESM2 MLM Training with DDP")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * world_size}")
        print("="*60)
    
    # Create trainer
    trainer = ESMTrainerDDP(config, rank, world_size, resume_from=None)
    
    # Train
    trainer.train()
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
