#!/bin/bash
# =============================================================================
# Brain MAE Installation Script
# Production-ready environment setup for 8x A100 GPU training
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Brain MAE Installation Script"
echo "=============================================="

# Configuration
ENV_NAME="${ENV_NAME:-brain_mae}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# =============================================================================
# Check prerequisites
# =============================================================================
echo ""
echo "📋 Checking prerequisites..."

# Check if conda is available
if command -v conda &> /dev/null; then
    print_status "Conda found: $(conda --version)"
    USE_CONDA=true
else
    print_warning "Conda not found, using pip/venv"
    USE_CONDA=false
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=true
else
    print_warning "No NVIDIA GPU detected, will install CPU version"
    HAS_GPU=false
fi

# =============================================================================
# Create environment
# =============================================================================
echo ""
echo "🔧 Creating environment..."

if [ "$USE_CONDA" = true ]; then
    # Conda environment
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment ${ENV_NAME} already exists"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n ${ENV_NAME} -y
        else
            print_status "Using existing environment"
            conda activate ${ENV_NAME}
        fi
    fi
    
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    fi
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    print_status "Activated conda environment: ${ENV_NAME}"
else
    # Python venv
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    print_status "Activated venv"
fi

# =============================================================================
# Install PyTorch
# =============================================================================
echo ""
echo "🔥 Installing PyTorch..."

if [ "$HAS_GPU" = true ]; then
    # GPU version with CUDA
    if [ "$USE_CONDA" = true ]; then
        conda install pytorch torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    # CPU version
    if [ "$USE_CONDA" = true ]; then
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
print_status "PyTorch installed"

# =============================================================================
# Install core dependencies
# =============================================================================
echo ""
echo "📦 Installing core dependencies..."

pip install --upgrade pip

# Core ML packages
pip install \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    pandas>=2.0.0

# Neuroimaging
pip install \
    nibabel>=5.0.0 \
    nilearn>=0.10.0

# Deep learning utilities
pip install \
    einops>=0.6.0 \
    timm>=0.9.0

# Wavelet transform
pip install PyWavelets>=1.4.0

# Visualization
pip install \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    plotly>=5.14.0

# Logging and experiment tracking
pip install \
    wandb>=0.15.0 \
    tensorboard>=2.13.0 \
    tqdm>=4.65.0

# Configuration and utilities
pip install \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    python-dotenv>=1.0.0

print_status "Core dependencies installed"

# =============================================================================
# Install DataLad for data downloading
# =============================================================================
echo ""
echo "📥 Installing DataLad..."

pip install datalad>=0.18.0

# Check git-annex (required by DataLad)
if ! command -v git-annex &> /dev/null; then
    print_warning "git-annex not found, attempting to install..."
    if [ "$USE_CONDA" = true ]; then
        conda install -c conda-forge git-annex -y
    else
        # Try apt-get (Debian/Ubuntu)
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-annex
        else
            print_error "Please install git-annex manually"
            print_error "  Ubuntu/Debian: sudo apt-get install git-annex"
            print_error "  macOS: brew install git-annex"
        fi
    fi
fi

print_status "DataLad installed"

# =============================================================================
# Install distributed training dependencies
# =============================================================================
echo ""
echo "🚀 Installing distributed training dependencies..."

if [ "$HAS_GPU" = true ]; then
    # DeepSpeed for efficient distributed training
    pip install deepspeed>=0.10.0
    
    # Flash Attention (for A100s)
    pip install flash-attn --no-build-isolation 2>/dev/null || \
        print_warning "Flash Attention installation failed (optional)"
    
    # NVIDIA Apex (optional, for fused optimizers)
    # pip install apex  # Requires compilation
fi

print_status "Distributed training dependencies installed"

# =============================================================================
# Install sentence-transformers for CLIP training
# =============================================================================
echo ""
echo "🤖 Installing sentence-transformers..."

pip install sentence-transformers>=2.2.0
pip install transformers>=4.30.0

print_status "Sentence transformers installed"

# =============================================================================
# Install development dependencies
# =============================================================================
echo ""
echo "🛠️ Installing development dependencies..."

pip install \
    pytest>=7.3.0 \
    pytest-cov>=4.1.0 \
    black>=23.3.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    mypy>=1.3.0 \
    pre-commit>=3.3.0

print_status "Development dependencies installed"

# =============================================================================
# Verify installation
# =============================================================================
echo ""
echo "🔍 Verifying installation..."

python << 'EOF'
import sys

def check_import(module, name=None):
    name = name or module
    try:
        __import__(module)
        print(f"  ✓ {name}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: {e}")
        return False

print("\nCore packages:")
all_ok = True
all_ok &= check_import("torch", "PyTorch")
all_ok &= check_import("numpy", "NumPy")
all_ok &= check_import("nibabel", "NiBabel")
all_ok &= check_import("pywt", "PyWavelets")
all_ok &= check_import("einops", "Einops")
all_ok &= check_import("wandb", "W&B")
all_ok &= check_import("sentence_transformers", "Sentence Transformers")

print("\nPyTorch details:")
import torch
print(f"  Version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

if not all_ok:
    print("\n⚠️ Some packages failed to import")
    sys.exit(1)
else:
    print("\n✓ All packages installed successfully!")
EOF

# =============================================================================
# Create requirements.txt
# =============================================================================
echo ""
echo "📄 Generating requirements.txt..."

pip freeze > requirements.txt
print_status "requirements.txt generated"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
if [ "$USE_CONDA" = true ]; then
    echo "  conda activate ${ENV_NAME}"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "To download data:"
echo "  python scripts/download_data.py --subjects 10 --tasks tunnel"
echo ""
echo "To run training:"
echo "  python train.py  # Single GPU"
echo "  torchrun --nproc_per_node=8 train.py  # 8 GPUs"
echo ""
echo "For more information, see README.md"
