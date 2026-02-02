"""
Configuration for Brain MAE (Masked Autoencoder) - Production Ready
Following expert principles: Data > Architecture > Hyperparameters

Scaling for multi-subject training with 8x A100 GPUs
"""


from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from pathlib import Path



@dataclass
class DataConfig:
    """Data configuration for multi-subject training"""
    # Paths
    bids_dir: Path = Path("/app/tmp/brain_llm/ds002345")
    output_dir: Path = Path("/app/tmp/brain_llm/BrainAutoencoder/outputs")
    checkpoint_dir: Path = Path("/app/tmp/brain_llm/BrainAutoencoder/checkpoints")
    
    # Multi-subject selection (production scale)
    # DS002345 has ~200 subjects, each with multiple tasks
    subjects: List[str] = field(default_factory=lambda: [])  # Populate dynamically from BIDS
    tasks: List[str] = field(default_factory=lambda: ["tunnel", "rest", "motor", "visual", "language"])
    
    # fMRI dimensions (from validation: 64x64x27x1040 per subject)
    volume_shape: Tuple[int, int, int] = (64, 64, 27)
    
    # Temporal windowing for more samples
    # Instead of 1040 volumes as separate samples, use sliding windows
    temporal_window: int = 16  # TRs per sample (like VideoMAE)
    temporal_stride: int = 4   # Overlap between windows
    
    # Train/Val/Test split (now across subjects, not time)
    train_subjects_ratio: float = 0.70
    val_subjects_ratio: float = 0.15
    test_subjects_ratio: float = 0.15
    
    random_seed: int = 42
    
    # Data loading
    cache_dir: Optional[Path] = None  # For caching preprocessed data
    preload_data: bool = False  # True if enough RAM



@dataclass
class PatchConfig:
    """3D Patch configuration - FIXED"""
    # Patch size: powers of 2 for GPU efficiency
    # (8, 8, 8) gives 8*8*4 = 256 patches per volume (27//8=3, but better to pad)
    # Alternative: (8, 8, 9) as before, or pad to 32 depth
    
    patch_size: Tuple[int, int, int] = (8, 8, 8)
    
    # Computed in __post_init__ - NO defaults to avoid inconsistency
    n_patches: int = field(init=False)
    patch_dim: int = field(init=False)
    grid_size: Tuple[int, int, int] = field(init=False)
    
    # Padding for depth (27 -> 32 to make divisible by 8)
    pad_depth: bool = True
    padded_shape: Tuple[int, int, int] = (64, 64, 32)
    
    def __post_init__(self):
        if self.pad_depth:
            effective_shape = self.padded_shape
        else:
            effective_shape = (64, 64, 27)
        
        self.n_patches = (
            (effective_shape[0] // self.patch_size[0]) * 
            (effective_shape[1] // self.patch_size[1]) * 
            (effective_shape[2] // self.patch_size[2])
        )
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.grid_size = (
            effective_shape[0] // self.patch_size[0],
            effective_shape[1] // self.patch_size[1], 
            effective_shape[2] // self.patch_size[2]
        )



@dataclass
class ModelConfig:
    """
    Production MAE Architecture
    
    With 200 subjects * 4 tasks * 1000 TRs / 16 window = ~50k samples
    Can scale to 25-50M params following Chinchilla
    """
    # Embedding dimension (ViT-Base scale for production)
    embed_dim: int = 768
    
    # Encoder (main learning happens here)
    encoder_depth: int = 12        # Standard ViT-Base
    encoder_heads: int = 12        # embed_dim must be divisible by heads
    encoder_mlp_ratio: float = 4.0  # Standard 4x expansion
    
    # Decoder (lightweight reconstruction)
    decoder_embed_dim: int = 512   # Smaller than encoder
    decoder_depth: int = 6         # Sufficient for reconstruction
    decoder_heads: int = 8
    
    # MAE masking
    mask_ratio: float = 0.75       # Standard MAE, optimal for 3D per BM-MAE [web:86]
    temporal_mask_ratio: float = 0.5  # Extra masking for temporal dimension (VideoMAE style)
    
    # Positional encoding
    use_3d_pos_embed: bool = True  # 3D sine-cosine positional encoding
    learnable_pos_embed: bool = False  # Sine-cosine works better for MAE
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path_rate: float = 0.1    # Stochastic depth for regularization
    
    # Gradient checkpointing (for memory efficiency with large models)
    use_gradient_checkpointing: bool = True
    
    # Parameter count: ~85M (ViT-Base scale)
    # Good for 50k+ samples (Chinchilla ratio ~0.0017)



@dataclass
class TrainingConfig:
    """Training hyperparameters for 8x A100 GPUs"""
    # Batch size: 32 per GPU * 8 GPUs = 256 global batch
    batch_size: int = 32           # Per GPU
    gradient_accumulation_steps: int = 1  # No need with large batch
    
    # Learning rate (scaled for large batch: sqrt(256/256) = 1x base)
    # Or use linear scaling: 1e-4 * (256/256) = 1e-4
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    
    # Schedule
    warmup_epochs: int = 10        # Longer warmup for large models
    epochs: int = 30              # Early stopping will cut this short
    
    # Optimizer (AdamW standard for MAE)
    optimizer: str = "adamw"
    weight_decay: float = 0.05     # Standard for MAE [web:86]
    betas: Tuple[float, float] = (0.9, 0.95)
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 30             # More patience for large models
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 10
    
    # Device
    device: str = "cuda"
    num_workers: int = 4           # For multi-GPU data loading
    
    # Mixed precision (BF16 for A100s - faster and no gradient scaling needed)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"    # A100 supports BF16 natively
    
    # Distributed training
    use_distributed: bool = True
    world_size: int = 8            # 8 GPUs
    dist_backend: str = "nccl"
    
    # Loss weights
    mse_loss_weight: float = 1.0
    perceptual_loss_weight: float = 0.0  # Can add later for better quality



@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Metrics to compute
    compute_mse: bool = True
    compute_mae: bool = True
    compute_ssim: bool = True
    compute_psnr: bool = True
    
    # Visualization
    n_samples_to_plot: int = 10
    plot_slices: List[str] = field(default_factory=lambda: ["axial", "sagittal", "coronal"])
    
    # Latent analysis
    run_pca: bool = True
    pca_components: int = 2
    run_tsne: bool = True
    
    # Reconstruction quality assessment
    compute_fid: bool = False      # FID for generated samples (if applicable)
    compute_inception: bool = False  # Inception score equivalent



@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Experiment name
    experiment_name: str = "brain_mae_production"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "brain_mae"
    log_interval: int = 100        # Steps between logs
    
    def __post_init__(self):
        # Create directories
        self.data.output_dir.mkdir(parents=True, exist_ok=True)
        self.data.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.data.cache_dir:
            self.data.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("BRAIN MAE CONFIGURATION (Production)")
        print("=" * 60)
        
        # Compute effective samples
        n_subjects = len(self.data.subjects) if self.data.subjects else 200
        est_samples = n_subjects * len(self.data.tasks) * (1000 // self.data.temporal_window)
        
        print(f"\n📊 Data:")
        print(f"  Subjects: {n_subjects} (estimated)")
        print(f"  Tasks: {self.data.tasks}")
        print(f"  Volume shape: {self.data.volume_shape}")
        print(f"  Temporal window: {self.data.temporal_window} (stride: {self.data.temporal_stride})")
        print(f"  Estimated samples: ~{est_samples:,}")
        
        print(f"\n🧩 Patches:")
        print(f"  Patch size: {self.patch.patch_size}")
        print(f"  Padded shape: {self.patch.padded_shape} (pad: {self.patch.pad_depth})")
        print(f"  Patches per volume: {self.patch.n_patches}")
        print(f"  Grid size: {self.patch.grid_size}")
        
        print(f"\n🏗️ Model (ViT-Base scale):")
        print(f"  Embed dim: {self.model.embed_dim}")
        print(f"  Encoder: {self.model.encoder_depth} layers, {self.model.encoder_heads} heads")
        print(f"  Decoder: {self.model.decoder_depth} layers, {self.model.decoder_heads} heads")
        print(f"  Mask ratio: {self.model.mask_ratio:.0%} (temporal: {self.model.temporal_mask_ratio:.0%})")
        print(f"  Total params: ~85M")
        
        print(f"\n⚙️ Training (8x A100):")
        print(f"  Global batch size: {self.training.batch_size * self.training.world_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Epochs: {self.training.epochs} (early stopping: {self.training.patience})")
        print(f"  AMP: {self.training.use_amp} ({self.training.amp_dtype})")
        print(f"  Gradient checkpointing: {self.model.use_gradient_checkpointing}")
        print("=" * 60)



def get_config() -> Config:
    """Get production configuration"""
    return Config()



def get_prototype_config() -> Config:
    """Get small config for debugging"""
    config = Config()
    
    # Small model for quick testing
    config.model.embed_dim = 128
    config.model.encoder_depth = 3
    config.model.encoder_heads = 4  # 128 / 4 = 32 (divisible)
    config.model.decoder_depth = 2
    config.model.decoder_embed_dim = 64
    config.model.decoder_heads = 4  # 64 / 4 = 16 (divisible)
    
    config.training.batch_size = 4
    config.training.world_size = 1
    config.training.use_amp = False
    config.training.device = "cpu"
    config.training.num_workers = 0
    
    config.data.subjects = ["sub-001"]
    config.data.tasks = ["tunnel"]
    
    return config



def get_single_gpu_config() -> Config:
    """Configuration for single A100 (80GB)"""
    config = Config()
    
    # Slightly smaller model for single GPU
    config.model.embed_dim = 512
    config.model.encoder_depth = 8
    
    config.training.batch_size = 16
    config.training.world_size = 1
    config.training.gradient_accumulation_steps = 2  # Effective batch 32
    
    return config



if __name__ == "__main__":
    config = get_config()
    config.print_summary()


# """
# Configuration for Brain MAE (Masked Autoencoder)
# Following expert principles: Data > Architecture > Hyperparameters

# Model sizing based on Chinchilla scaling laws:
# - 1040 volumes (sub-001 tunnel task)
# - Target params: ~1-5M (not 100M+!)
# - Params/Data ratio < 0.01
# """

# from dataclasses import dataclass, field
# from typing import Tuple, List, Optional
# from pathlib import Path


# @dataclass
# class DataConfig:
#     """Data configuration"""
#     # Paths
#     bids_dir: Path = Path("/app/tmp/brain_llm/ds002345")
#     output_dir: Path = Path("/app/tmp/brain_llm/BrainAutoencoder/outputs")
#     checkpoint_dir: Path = Path("/app/tmp/brain_llm/BrainAutoencoder/checkpoints")
    
#     # Subject selection (start with one for prototype)
#     subjects: List[str] = field(default_factory=lambda: ["sub-001"])
#     tasks: List[str] = field(default_factory=lambda: ["tunnel"])
    
#     # fMRI dimensions (from validation: 64x64x27x1040)
#     volume_shape: Tuple[int, int, int] = (64, 64, 27)
    
#     # Train/Val split
#     train_ratio: float = 0.85  # 900 TRs train, 140 TRs val
#     random_seed: int = 42


# @dataclass
# class PatchConfig:
#     """3D Patch configuration"""
#     # Patch size must divide volume evenly
#     # 64/8=8, 64/8=8, 27/9=3 → 8*8*3 = 192 patches per volume
#     patch_size: Tuple[int, int, int] = (8, 8, 9)
    
#     # Derived values (computed in __post_init__)
#     n_patches: int = 192  # 8*8*3
#     patch_dim: int = 576  # 8*8*9 = 576 voxels per patch
#     grid_size: Tuple[int, int, int] = (8, 8, 3)
    
#     def __post_init__(self):
#         self.n_patches = (64 // self.patch_size[0]) * (64 // self.patch_size[1]) * (27 // self.patch_size[2])
#         self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
#         self.grid_size = (64 // self.patch_size[0], 64 // self.patch_size[1], 27 // self.patch_size[2])


# @dataclass
# class WaveletConfig:
#     """Wavelet transform configuration"""
#     enabled: bool = True
#     wavelet: str = "db1"  # Daubechies-1 (Haar)
#     level: int = 1  # Single level decomposition (simpler for prototype)
#     # With level=1: 8 coefficient sets (1 approx + 7 detail)


# @dataclass
# class ModelConfig:
#     """
#     Model architecture - SMALL for prototype!
    
#     Expert rule: params/data < 0.01
#     With 1040 samples: max ~10k params ideal, ~1M acceptable with regularization
    
#     Reduced from 256->128 embed_dim for better generalization on small data.
#     """
#     # Embedding dimension (reduced for small dataset)
#     embed_dim: int = 128  # Reduced from 256
    
#     # Encoder (main learning happens here)
#     encoder_depth: int = 3  # Reduced from 4
#     encoder_heads: int = 4
#     encoder_mlp_ratio: float = 2.0  # FFN hidden = embed_dim * mlp_ratio
    
#     # Decoder (lightweight reconstruction)
#     decoder_embed_dim: int = 64  # Reduced from 128
#     decoder_depth: int = 2  # Very shallow
#     decoder_heads: int = 4
    
#     # MAE masking
#     mask_ratio: float = 0.75  # Mask 75% of patches (standard MAE)
    
#     # Regularization
#     dropout: float = 0.1
#     attention_dropout: float = 0.1
    
#     # Approximate param count (reduced):
#     # Patch embed: 576 * 128 = 74k
#     # Encoder: 3 * (128^2 * 4 + 128 * 256 * 2) ≈ 400k
#     # Decoder: 2 * (64^2 * 4 + 64 * 128 * 2) ≈ 65k
#     # Total: ~600k params (better for 1040 samples!)


# @dataclass
# class TrainingConfig:
#     """Training hyperparameters - following expert guidelines"""
#     # Batch size (small for CPU, can increase for GPU)
#     batch_size: int = 4  # Start small, increase if memory allows
    
#     # Learning rate (will use LR finder to tune)
#     learning_rate: float = 1e-4  # Conservative start
#     min_lr: float = 1e-6
    
#     # Schedule
#     warmup_epochs: int = 5
#     epochs: int = 100
    
#     # Optimizer
#     optimizer: str = "adamw"
#     weight_decay: float = 0.05  # Strong regularization for small data
#     betas: Tuple[float, float] = (0.9, 0.95)
    
#     # Gradient clipping
#     max_grad_norm: float = 1.0
    
#     # Early stopping
#     patience: int = 15  # Stop if no improvement for 15 epochs
#     min_delta: float = 1e-4
    
#     # Checkpointing
#     save_every: int = 10
    
#     # Device
#     device: str = "cpu"  # Will auto-detect GPU
#     num_workers: int = 0  # For CPU, use 0
    
#     # Mixed precision (for GPU)
#     use_amp: bool = False  # Enable on GPU


# @dataclass
# class EvalConfig:
#     """Evaluation configuration"""
#     # Metrics to compute
#     compute_mse: bool = True
#     compute_mae: bool = True
#     compute_ssim: bool = True
#     compute_psnr: bool = True
    
#     # Visualization
#     n_samples_to_plot: int = 5
#     plot_slices: List[str] = field(default_factory=lambda: ["axial", "sagittal", "coronal"])
    
#     # Latent analysis
#     run_pca: bool = True
#     pca_components: int = 2


# @dataclass
# class Config:
#     """Master configuration"""
#     data: DataConfig = field(default_factory=DataConfig)
#     patch: PatchConfig = field(default_factory=PatchConfig)
#     wavelet: WaveletConfig = field(default_factory=WaveletConfig)
#     model: ModelConfig = field(default_factory=ModelConfig)
#     training: TrainingConfig = field(default_factory=TrainingConfig)
#     eval: EvalConfig = field(default_factory=EvalConfig)
    
#     # Experiment name
#     experiment_name: str = "brain_mae_prototype"
    
#     def __post_init__(self):
#         # Create directories
#         self.data.output_dir.mkdir(parents=True, exist_ok=True)
#         self.data.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
#     def print_summary(self):
#         """Print configuration summary"""
#         print("=" * 60)
#         print("BRAIN MAE CONFIGURATION")
#         print("=" * 60)
#         print(f"\n📊 Data:")
#         print(f"  Subjects: {self.data.subjects}")
#         print(f"  Tasks: {self.data.tasks}")
#         print(f"  Volume shape: {self.data.volume_shape}")
#         print(f"  Train/Val split: {self.data.train_ratio:.0%}/{1-self.data.train_ratio:.0%}")
        
#         print(f"\n🧩 Patches:")
#         print(f"  Patch size: {self.patch.patch_size}")
#         print(f"  Patches per volume: {self.patch.n_patches}")
#         print(f"  Patch dimension: {self.patch.patch_dim}")
        
#         print(f"\n🌊 Wavelet:")
#         print(f"  Enabled: {self.wavelet.enabled}")
#         print(f"  Type: {self.wavelet.wavelet}, Level: {self.wavelet.level}")
        
#         print(f"\n🏗️ Model:")
#         print(f"  Embed dim: {self.model.embed_dim}")
#         print(f"  Encoder: {self.model.encoder_depth} layers, {self.model.encoder_heads} heads")
#         print(f"  Decoder: {self.model.decoder_depth} layers, {self.model.decoder_heads} heads")
#         print(f"  Mask ratio: {self.model.mask_ratio:.0%}")
        
#         print(f"\n⚙️ Training:")
#         print(f"  Batch size: {self.training.batch_size}")
#         print(f"  Learning rate: {self.training.learning_rate}")
#         print(f"  Epochs: {self.training.epochs}")
#         print(f"  Weight decay: {self.training.weight_decay}")
#         print(f"  Early stopping patience: {self.training.patience}")
#         print("=" * 60)


# def get_config() -> Config:
#     """Get default configuration"""
#     return Config()


# def get_gpu_config() -> Config:
#     """Configuration optimized for 8x A100 GPUs"""
#     config = Config()
    
#     # Scale up for multi-GPU
#     config.training.batch_size = 32  # Per GPU
#     config.training.num_workers = 4
#     config.training.use_amp = True
#     config.training.device = "cuda"
    
#     # Can use larger model with more data
#     config.model.embed_dim = 512
#     config.model.encoder_depth = 8
#     config.model.decoder_depth = 4
    
#     return config


# if __name__ == "__main__":
#     config = get_config()
#     config.print_summary()
