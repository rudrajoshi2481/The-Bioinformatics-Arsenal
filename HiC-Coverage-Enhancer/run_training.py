#!/usr/bin/env python3
"""
Hi-C Enhancement Training Runner
=================================

Script to run training with the modular pipeline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from trainer import train_model
from dataset import split_dataset


# =============================================================================
# CONFIGURATION - Edit these settings for your training
# =============================================================================

# Data paths (output from preprocessing.py)
DATA_DIR = './training_data'

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 16          # Reduce if out of memory
LEARNING_RATE = 2e-4
NUM_WORKERS = 2          # Data loading workers

# Model architecture
BASE_CHANNELS = 32       # Base number of channels
NUM_TRANSFORMER_LAYERS = 2
NUM_HEADS = 4

# Output directory for experiments
OUTPUT_DIR = './experiments/hic_enhancement'

# =============================================================================


def main():
    """Main training function"""
    
    config = Config()
    
    # Paths
    train_path = f'{DATA_DIR}/train_dataset.npz'
    val_path = f'{DATA_DIR}/val_dataset.npz'
    
    # Check if we need to split the dataset
    if not os.path.exists(val_path) and os.path.exists(train_path):
        print("Splitting dataset into train/val/test...")
        train_path, val_path, test_path = split_dataset(
            train_path,
            DATA_DIR,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )
    
    # Update config from settings
    config.data.train_data = train_path
    config.data.val_data = val_path
    config.output_dir = OUTPUT_DIR
    
    config.training.num_epochs = NUM_EPOCHS
    config.training.batch_size = BATCH_SIZE
    config.training.learning_rate = LEARNING_RATE
    config.training.num_workers = NUM_WORKERS
    config.training.save_every = 5
    config.training.plot_every = 1
    
    config.model.base_channels = BASE_CHANNELS
    config.model.num_transformer_layers = NUM_TRANSFORMER_LAYERS
    config.model.num_heads = NUM_HEADS
    
    print("\n" + "="*70)
    print("HI-C ENHANCEMENT TRAINING")
    print("="*70)
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print("="*70 + "\n")
    
    # Run training
    exp_dir = train_model(
        train_data=train_path,
        val_data=val_path if os.path.exists(val_path) else None,
        config=config
    )
    
    print(f"\nTraining complete! Results saved to: {exp_dir}")
    print(f"\nNext step: Generate enhanced mcool file:")
    print(f"  python inference.py --model {exp_dir}/best_model.pth --input your_data.mcool --output enhanced.mcool")


if __name__ == "__main__":
    main()
