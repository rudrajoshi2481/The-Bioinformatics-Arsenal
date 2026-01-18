"""
Hi-C Enhancement Configuration
==============================

Centralized configuration for the Hi-C enhancement pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json
import os


@dataclass
class DataConfig:
    """Data-related configuration"""
    train_data: str = './training_data/train_dataset.npz'
    val_data: str = './training_data/val_dataset.npz'
    chunk_size: int = 40
    stride: int = 28
    downsample_ratio: int = 16
    high_cutoff: int = 255
    low_cutoff: int = 100
    min_contacts: int = 5
    

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    base_channels: int = 64
    num_transformer_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    augment: bool = True
    save_every: int = 5
    plot_every: int = 1
    l1_weight: float = 0.7
    l2_weight: float = 0.3


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    mcool_files: List[str] = field(default_factory=list)
    resolutions: List[int] = field(default_factory=lambda: [10000])
    output_dir: str = './training_data'
    max_memory_gb: float = 2.0
    use_all_chroms: bool = False
    train_chroms: List[str] = field(default_factory=lambda: [
        f'chr{i}' for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    ])
    val_chroms: List[str] = field(default_factory=lambda: ['chr17', 'chr21'])
    test_chroms: List[str] = field(default_factory=lambda: ['chr22'])


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = ''
    input_mcool: str = ''
    output_mcool: str = './enhanced_output.mcool'
    resolution: int = 10000
    batch_size: int = 64
    chromosomes: List[str] = field(default_factory=lambda: [f'chr{i}' for i in range(1, 23)])


@dataclass
class Config:
    """Main configuration container"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: str = './experiments/hic_enhancement'
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    seed: int = 42
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'inference': self.inference.__dict__,
            'output_dir': self.output_dir,
            'device': self.device,
            'seed': self.seed
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'preprocessing' in config_dict:
            config.preprocessing = PreprocessingConfig(**config_dict['preprocessing'])
        if 'inference' in config_dict:
            config.inference = InferenceConfig(**config_dict['inference'])
        if 'output_dir' in config_dict:
            config.output_dir = config_dict['output_dir']
        if 'device' in config_dict:
            config.device = config_dict['device']
        if 'seed' in config_dict:
            config.seed = config_dict['seed']
        
        return config
    
    def get_device(self):
        """Get the appropriate torch device"""
        import torch
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def create_experiment_dir(base_dir: str) -> str:
    """Create timestamped experiment directory"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"{base_dir}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
