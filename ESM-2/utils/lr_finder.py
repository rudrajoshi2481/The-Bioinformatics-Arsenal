# lr_finder.py
"""
Learning Rate Finder for ESM Training
Based on Leslie Smith's 2015 paper
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import sys
import os


class LRFinder:
    """Find optimal learning rate using range test"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=1e-2, num_iter=200):
        """
        Test range of learning rates
        
        Args:
            train_loader: DataLoader for training data
            start_lr: Starting learning rate (very small)
            end_lr: Ending learning rate (large)
            num_iter: Number of iterations to test
        """
        print("=" * 70)
        print("🔍 LEARNING RATE FINDER")
        print("=" * 70)
        print(f"Testing LR range: {start_lr:.2e} to {end_lr:.2e}")
        print(f"Number of iterations: {num_iter}")
        print("=" * 70)
        
        # Save initial state
        initial_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        initial_opt_state = self.optimizer.state_dict()
        
        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        
        self.optimizer.param_groups[0]['lr'] = lr
        
        self.model.train()
        smoothed_loss = 0
        best_loss = float('inf')
        batch_iter = iter(train_loader)
        
        for iteration in range(num_iter):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # Compute smoothed loss
            if iteration == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss.item()
            
            # Record
            self.history['lr'].append(lr)
            self.history['loss'].append(smoothed_loss)
            
            # Check for divergence
            if smoothed_loss > 4 * best_loss or torch.isnan(loss):
                print(f"\n⚠️  Loss exploded at LR={lr:.2e}")
                print(f"   Current loss: {smoothed_loss:.4f}")
                print(f"   Best loss: {best_loss:.4f}")
                print(f"   Stopping early at iteration {iteration}/{num_iter}")
                break
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update LR
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr
            
            # Progress
            if iteration % 20 == 0:
                print(f"Iter {iteration:3d}/{num_iter} | LR: {lr:.2e} | Loss: {smoothed_loss:.4f} | Best: {best_loss:.4f}")
        
        print("\n✅ LR Finder test complete!")
        
        # Restore initial state
        self.model.load_state_dict(initial_model_state)
        self.optimizer.load_state_dict(initial_opt_state)
        
        return self.history
    
    def plot(self, skip_start=10, skip_end=5, save_path='lr_finder_plot.png'):
        """Plot and analyze results"""
        if len(self.history['lr']) < skip_start + skip_end:
            skip_start = 0
            skip_end = 0
        
        lrs = np.array(self.history['lr'][skip_start:-skip_end if skip_end > 0 else None])
        losses = np.array(self.history['loss'][skip_start:-skip_end if skip_end > 0 else None])
        
        # Find optimal LR (steepest gradient)
        try:
            # Use gradient method
            gradients = np.gradient(losses)
            min_grad_idx = gradients.argmin()
            
            # Find minimum loss
            min_loss_idx = losses.argmin()
            
            # Optimal is usually before minimum (where gradient is steepest)
            optimal_lr = lrs[min_grad_idx]
            min_loss_lr = lrs[min_loss_idx]
            
        except:
            optimal_lr = lrs[len(lrs)//2]
            min_loss_lr = lrs[losses.argmin()]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Loss vs LR (log scale)
        ax1.plot(lrs, losses, linewidth=2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Learning Rate Finder: Loss vs LR', fontsize=14, fontweight='bold')
        ax1.axvline(optimal_lr, color='red', linestyle='--', linewidth=2,
                   label=f'Suggested LR: {optimal_lr:.2e}')
        ax1.axvline(min_loss_lr, color='green', linestyle=':', linewidth=2,
                   label=f'Min Loss LR: {min_loss_lr:.2e}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Gradient
        try:
            gradients = np.gradient(losses)
            ax2.plot(lrs, gradients, linewidth=2, color='orange')
            ax2.set_xscale('log')
            ax2.set_xlabel('Learning Rate', fontsize=12)
            ax2.set_ylabel('Loss Gradient', fontsize=12)
            ax2.set_title('Loss Gradient (Steepest = Best)', fontsize=14, fontweight='bold')
            ax2.axvline(optimal_lr, color='red', linestyle='--', linewidth=2,
                       label=f'Steepest: {optimal_lr:.2e}')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        except:
            ax2.text(0.5, 0.5, 'Gradient calculation failed', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: {save_path}")
        
        # Print recommendations
        print("\n" + "=" * 70)
        print("🎯 LEARNING RATE RECOMMENDATIONS")
        print("=" * 70)
        print(f"✅ Suggested LR (steepest gradient): {optimal_lr:.6f}")
        print(f"   Use this as your target learning rate")
        print(f"\n💡 Conservative LR (safer):         {optimal_lr/10:.6f}")
        print(f"   Use this if training is unstable")
        print(f"\n📍 Min loss LR:                      {min_loss_lr:.6f}")
        print(f"   (Usually too high, causes instability)")
        print("=" * 70)
        print(f"\n🔧 Update your train_config.py:")
        print(f"   self.learning_rate = {optimal_lr:.6f}")
        print(f"   self.warmup_steps = 2000")
        print("=" * 70)
        
        return optimal_lr


def find_optimal_lr():
    """Main function to run LR finder"""
    
    print("\n🚀 Starting LR Finder for ESM Training\n")
    
    # Import your modules
    from config import MLMConfig
    from dataset import MLMDataset
    from model import ESMForMaskedLM
    from config import get_esm_small_config, get_esm_medium_config
    from token import EsmTokenizer
    from torch.utils.data import DataLoader
    
    # Setup
    config = MLMConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Loading tokenizer...")
    
    # Get tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    print(f"Loading model...")
    
    # Create model config based on model_type
    if config.model_type == "small":
        model_config = get_esm_small_config()
        print("Using SMALL model (~8M params)")
    elif config.model_type == "medium":
        model_config = get_esm_medium_config()
        print("Using MEDIUM model (~35M params)")
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Create model
    model = ESMForMaskedLM(
        model_config,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loading data...")
    
    # Load dataset
    train_dataset = MLMDataset(
        json_file=config.train_json,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        mlm_prob=config.mlm_probability
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Batch size: {config.batch_size}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-7,  # Start very low
        weight_decay=config.weight_decay
    )
    
    # Run LR finder
    lr_finder = LRFinder(model, optimizer, device)
    history = lr_finder.range_test(
        train_loader,
        start_lr=1e-7,
        end_lr=1e-2,
        num_iter=200
    )
    
    # Plot and get optimal LR
    optimal_lr = lr_finder.plot(
        skip_start=10,
        skip_end=5,
        save_path='lr_finder_plot.png'
    )
    
    return optimal_lr


if __name__ == "__main__":
    try:
        optimal_lr = find_optimal_lr()
        print(f"\n✅ SUCCESS! Recommended LR: {optimal_lr:.6f}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
