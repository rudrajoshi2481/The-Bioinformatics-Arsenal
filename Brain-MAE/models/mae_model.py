"""
Masked Autoencoder (MAE) for 3D Brain fMRI
Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
Adapted for 3D brain volumes with optional wavelet preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, get_config
from models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    SinusoidalPositionalEncoding3D,
    LearnablePositionalEncoding3D,
    count_parameters
)


class PatchEmbedding(nn.Module):
    """Project flattened 3D patches to embedding dimension"""
    
    def __init__(self, patch_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, patch_dim)
        Returns:
            (batch, n_patches, embed_dim)
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class BrainMAE(nn.Module):
    """
    Masked Autoencoder for 3D Brain fMRI
    
    Architecture:
    1. Patch embedding: (B, n_patches, patch_dim) -> (B, n_patches, embed_dim)
    2. Random masking: Keep only (1 - mask_ratio) patches
    3. Encoder: Process visible patches with transformer
    4. Decoder: Reconstruct all patches (visible + masked)
    5. Reconstruction head: Project back to patch dimension
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.n_patches = config.patch.n_patches
        self.patch_dim = config.patch.patch_dim
        self.embed_dim = config.model.embed_dim
        self.decoder_embed_dim = config.model.decoder_embed_dim
        self.mask_ratio = config.model.mask_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(self.patch_dim, self.embed_dim)
        
        # Positional encoding (learnable works better for small datasets)
        self.pos_embed = LearnablePositionalEncoding3D(self.n_patches, self.embed_dim)
        
        # Encoder
        self.encoder = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=config.model.encoder_depth,
            num_heads=config.model.encoder_heads,
            mlp_ratio=config.model.encoder_mlp_ratio,
            dropout=config.model.dropout,
            attention_dropout=config.model.attention_dropout
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = LearnablePositionalEncoding3D(
            self.n_patches, self.decoder_embed_dim
        )
        
        self.decoder = TransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth=config.model.decoder_depth,
            num_heads=config.model.decoder_heads,
            mlp_ratio=config.model.encoder_mlp_ratio,
            dropout=config.model.dropout,
            attention_dropout=config.model.attention_dropout
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(self.decoder_embed_dim, self.patch_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following MAE paper"""
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def random_masking(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.
        
        Args:
            x: (batch, n_patches, embed_dim)
            mask_ratio: Fraction of patches to mask
        
        Returns:
            x_masked: Visible patches only (batch, n_visible, embed_dim)
            mask: Binary mask (batch, n_patches), 1 = masked
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first n_keep patches (after shuffling)
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Create binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, 
        x: torch.Tensor, 
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches.
        
        Args:
            x: (batch, n_patches, patch_dim)
            mask_ratio: Override default mask ratio
        
        Returns:
            latent: Encoded visible patches
            mask: Binary mask
            ids_restore: Indices to restore order
        """
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Encode visible patches
        latent = self.encoder(x)
        
        return latent, mask, ids_restore
    
    def forward_decoder(
        self, 
        latent: torch.Tensor, 
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode and reconstruct all patches.
        
        Args:
            latent: Encoded visible patches (batch, n_visible, embed_dim)
            ids_restore: Indices to restore original order
        
        Returns:
            reconstruction: (batch, n_patches, patch_dim)
        """
        # Project to decoder dimension
        x = self.decoder_embed(latent)
        
        # Append mask tokens
        B, N_vis, D = x.shape
        N_mask = self.n_patches - N_vis
        
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x = torch.gather(
            x, dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Add decoder positional encoding
        x = self.decoder_pos_embed(x)
        
        # Decode
        x = self.decoder(x)
        
        # Reconstruct patches
        reconstruction = self.reconstruction_head(x)
        
        return reconstruction
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: (batch, n_patches, patch_dim)
            mask_ratio: Override default mask ratio
        
        Returns:
            Dictionary with:
                - reconstruction: (batch, n_patches, patch_dim)
                - mask: Binary mask (batch, n_patches)
                - latent: Encoded representation
        """
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        reconstruction = self.forward_decoder(latent, ids_restore)
        
        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "latent": latent,
            "ids_restore": ids_restore
        }
    
    def compute_loss(
        self, 
        reconstruction: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor,
        loss_on_all: bool = False
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstruction: (batch, n_patches, patch_dim)
            target: (batch, n_patches, patch_dim)
            mask: Binary mask (batch, n_patches), 1 = masked
            loss_on_all: If True, compute loss on all patches (not just masked)
        
        Returns:
            MSE loss
        """
        if loss_on_all:
            # Loss on all patches
            loss = F.mse_loss(reconstruction, target)
        else:
            # Loss only on masked patches (standard MAE)
            # Expand mask to match patch dimensions
            mask = mask.unsqueeze(-1).expand_as(reconstruction)
            
            # Compute loss only where mask == 1
            loss = (reconstruction - target) ** 2
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without masking (for evaluation).
        
        Args:
            x: (batch, n_patches, patch_dim)
        
        Returns:
            latent: (batch, n_patches, embed_dim)
        """
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        latent = self.encoder(x)
        return latent


def create_model(config: Config) -> BrainMAE:
    """Create BrainMAE model from config"""
    model = BrainMAE(config)
    
    # Print model summary
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print("BRAIN MAE MODEL")
    print('='*60)
    print(f"Total parameters: {n_params:,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    print(f"  Patch embed: {count_parameters(model.patch_embed):,}")
    print(f"  Reconstruction head: {count_parameters(model.reconstruction_head):,}")
    
    # Check params/data ratio
    n_samples = 1040  # sub-001 tunnel task
    ratio = n_params / n_samples
    print(f"\nParams/Data ratio: {ratio:.2f}")
    if ratio > 100:
        print("⚠️  Warning: Model may be too large for dataset!")
    else:
        print("✓ Model size appropriate for dataset")
    
    return model


if __name__ == "__main__":
    # Test MAE model
    print("Testing Brain MAE Model")
    print("=" * 60)
    
    config = get_config()
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.patch.n_patches, config.patch.patch_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Mask shape: {output['mask'].shape}")
    print(f"Latent shape: {output['latent'].shape}")
    
    # Test loss computation
    loss = model.compute_loss(output['reconstruction'], x, output['mask'])
    print(f"\nReconstruction loss: {loss.item():.4f}")
    
    # Test without masking
    latent = model.get_latent_representation(x)
    print(f"Full latent shape: {latent.shape}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"Gradient norms (first 5 layers):")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    
    # Check for dead gradients
    dead_layers = [n for n, g in grad_norms if g < 1e-7]
    if dead_layers:
        print(f"\n⚠️  Dead layers (no gradients): {dead_layers}")
    else:
        print("\n✓ All layers receiving gradients")
    
    print("\n" + "=" * 60)
    print("Brain MAE model test complete!")
