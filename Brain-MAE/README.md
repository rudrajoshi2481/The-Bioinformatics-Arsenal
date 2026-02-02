# Brain MAE: Masked Autoencoder for fMRI

Production-ready 3D Masked Autoencoder for fMRI data, designed for CLIP-style cross-modal training with sentence-transformers.

## Project Structure

```
BrainAutoencoder/
├── configs/
│   └── config.py          # All hyperparameters (prototype & production)
├── data/
│   ├── preprocessing.py   # fMRI loading, normalization, patching
│   └── preprocessing_v2.py # Production version with padding
├── models/
│   ├── transformer.py     # Transformer blocks, attention, positional encoding
│   └── mae_model.py       # Main MAE architecture
├── validation/
│   ├── data_validation.py # Data quality checks
│   └── model_validation.py # Model sanity checks
├── scripts/
│   ├── download_data.py   # Multi-subject data downloader
│   └── installation.sh    # Environment setup script
├── checkpoints/           # Saved model weights
├── outputs/               # Results, plots, metrics
├── train.py               # Training script
├── evaluate.py            # Evaluation and visualization
└── README.md
```

## Installation

### Quick Install
```bash
cd /app/tmp/brain_llm/BrainAutoencoder
chmod +x scripts/installation.sh
./scripts/installation.sh
```

### Manual Install
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel nilearn einops timm PyWavelets wandb sentence-transformers
pip install datalad  # For data downloading
```

## Downloading Data

The Narratives dataset (ds002345) contains 345 subjects with multiple story-listening tasks.

### List Available Data
```bash
python scripts/download_data.py --list
```

### Download Specific Subjects
```bash
# Download first 10 subjects, tunnel task only
python scripts/download_data.py --subjects 10 --tasks tunnel

# Download specific subjects
python scripts/download_data.py --subject-ids sub-001 sub-002 sub-003 --tasks tunnel pieman

# Dry run (see what would be downloaded)
python scripts/download_data.py --dry-run --subjects 50 --tasks tunnel
```

### Download All Data (Warning: ~500GB)
```bash
python scripts/download_data.py --all
```

## Quick Start

### 1. Validate Data
```bash
python -m validation.data_validation
```

### 2. Validate Model
```bash
python -m validation.model_validation
```

### 3. Train (Prototype - Single Subject)
```bash
python train.py  # Uses prototype config by default
```

### 4. Train (Production - 8x A100 GPUs)
```bash
torchrun --nproc_per_node=8 train.py --config production
```

### 5. Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Architecture

### Model Design (Following Expert Principles)

**Key principle**: `params/data ratio < 100` for small datasets

| Component | Configuration | Parameters |
|-----------|--------------|------------|
| Patch size | 8×8×9 | - |
| Patches per volume | 192 | - |
| Embedding dim | 256 | - |
| Encoder | 4 layers, 4 heads | ~800K |
| Decoder | 2 layers, 4 heads | ~200K |
| **Total** | - | **~1.7M** |

With 1040 samples (sub-001 tunnel task), this gives a params/data ratio of ~1600, which is acceptable with strong regularization.

### MAE Pipeline

```
Input: 3D Volume (64×64×27)
    ↓
Extract Patches (192 patches of 8×8×9)
    ↓
Patch Embedding (576 → 256 dim)
    ↓
Add Positional Encoding (learnable 3D)
    ↓
Random Masking (75% masked)
    ↓
Transformer Encoder (4 layers)
    ↓
Latent Representation (48 visible patches × 256 dim)
    ↓
Append Mask Tokens + Unshuffle
    ↓
Transformer Decoder (2 layers)
    ↓
Reconstruction Head (128 → 576 dim)
    ↓
Output: Reconstructed Patches → Volume
```

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 4 | CPU-friendly, increase for GPU |
| Learning rate | 1e-4 | Conservative, use LR finder |
| Weight decay | 0.05 | Strong regularization for small data |
| Mask ratio | 75% | Standard MAE setting |
| Epochs | 100 | With early stopping |
| Warmup | 5 epochs | Prevents early instability |
| Schedule | Cosine annealing | Gold standard |

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MSE | < 0.05 | Mean squared error (normalized) |
| SSIM | > 0.85 | Structural similarity |
| PSNR | > 25 dB | Peak signal-to-noise ratio |
| Correlation | > 0.90 | Pearson correlation |

## Configuration Options

### Prototype (Single Subject, CPU/Single GPU)
```python
from configs import get_prototype_config
config = get_prototype_config()
# embed_dim=128, encoder_depth=3, batch_size=4
```

### Single A100 GPU
```python
from configs import get_single_gpu_config
config = get_single_gpu_config()
# embed_dim=512, encoder_depth=8, batch_size=16
```

### Production (8× A100 GPUs)
```python
from configs import get_config  # Default is production
config = get_config()
# embed_dim=768, encoder_depth=12, batch_size=32 per GPU
```

## Production Training (8× A100)

```bash
# Set up distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Launch with torchrun
torchrun --nproc_per_node=8 train.py

# Or with SLURM
srun --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 python train.py
```

## CLIP Training (Future)

This MAE is designed to be compatible with sentence-transformers/all-MiniLM-L6-v2 (384-dim output).

```python
# After MAE pretraining, freeze encoder and add projection
from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# text_encoder outputs 384-dim embeddings
# Brain encoder should project to 384-dim for CLIP alignment
```

## Expert Principles Applied

1. **Data > Architecture > Hyperparameters**
   - Comprehensive data validation before training
   - Model sized appropriately for data

2. **Always run sanity checks**
   - Single batch overfit test
   - Gradient flow verification
   - Patch reconstruction verification

3. **Proper training practices**
   - Cosine annealing with warmup
   - Early stopping
   - Gradient clipping
   - Mixed precision (BF16 on A100)

4. **Modular design**
   - Separate files for config, data, models, validation
   - Easy to extend and modify
