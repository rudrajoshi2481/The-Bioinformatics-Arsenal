import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Evoformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IPA'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from Evoformer import Evoformer
from structure_modult import StructureModule
from geometry import Rigid
from heads import PredictedLDDTHead, PredictedAlignedErrorHead, DistogramHead, MaskedMSAHead
from losses import AlphaFoldLoss


class InputEmbedding(nn.Module):
    def __init__(self, d_msa_input: int = 49, d_pair_input: int = 506, d_msa: int = 256, d_pair: int = 128):
        super().__init__()
        self.msa_embed = nn.Linear(d_msa_input, d_msa)
        self.pair_embed = nn.Linear(d_pair_input, d_pair)
        self.msa_norm = nn.LayerNorm(d_msa)
        self.pair_norm = nn.LayerNorm(d_pair)

    def forward(self, msa_feat: torch.Tensor, pair_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        msa_emb = self.msa_norm(self.msa_embed(msa_feat))
        pair_emb = self.pair_norm(self.pair_embed(pair_feat))
        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    def __init__(self, d_msa: int = 256, d_pair: int = 128, num_bins: int = 15, min_bin: float = 3.25, max_bin: float = 20.75):
        super().__init__()
        self.d_msa = d_msa
        self.d_pair = d_pair
        self.num_bins = num_bins

        self.msa_norm = nn.LayerNorm(d_msa)
        self.pair_norm = nn.LayerNorm(d_pair)

        self.dgram_linear = nn.Linear(num_bins, d_pair)

        self.register_buffer('min_bin', torch.tensor(min_bin))
        self.register_buffer('max_bin', torch.tensor(max_bin))

    def compute_distogram(self, positions: torch.Tensor) -> torch.Tensor:
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

        bin_width = (self.max_bin - self.min_bin) / self.num_bins
        lower_bounds = self.min_bin + bin_width * torch.arange(self.num_bins, device=positions.device)

        dgram = (distances.unsqueeze(-1) > lower_bounds).float()
        dgram = dgram[..., :-1] - dgram[..., 1:]
        dgram = torch.cat([dgram, (distances.unsqueeze(-1) > lower_bounds[-1]).float()], dim=-1)

        return dgram

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        prev_msa: Optional[torch.Tensor],
        prev_pair: Optional[torch.Tensor],
        prev_positions: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_msa is not None:
            msa_update = self.msa_norm(prev_msa[:, 0:1])
            msa = msa.clone()
            msa[:, 0:1] = msa[:, 0:1] + msa_update

        if prev_pair is not None:
            pair = pair + self.pair_norm(prev_pair)

        if prev_positions is not None:
            dgram = self.compute_distogram(prev_positions)
            pair = pair + self.dgram_linear(dgram)

        return msa, pair


class AlphaFold2(nn.Module):
    def __init__(
        self,
        d_msa: int = 256,
        d_pair: int = 128,
        d_single: int = 384,
        num_evoformer_blocks: int = 48,
        num_structure_layers: int = 8,
        num_heads: int = 8,
        num_recycles: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_msa = d_msa
        self.d_pair = d_pair
        self.d_single = d_single
        self.num_recycles = num_recycles

        self.input_embedding = InputEmbedding(d_msa_input=49, d_pair_input=506, d_msa=d_msa, d_pair=d_pair)

        self.recycling_embedder = RecyclingEmbedder(d_msa=d_msa, d_pair=d_pair)

        self.evoformer = Evoformer(d_msa=d_msa, d_pair=d_pair, num_blocks=num_evoformer_blocks,
                                   num_heads=num_heads, dropout=dropout)

        self.msa_to_single = nn.Linear(d_msa, d_single)

        self.structure_module = StructureModule(d_single=d_single, d_pair=d_pair,
                                                num_layers=num_structure_layers, num_heads=12, dropout=dropout)

        self.plddt_head = PredictedLDDTHead(d_single, num_bins=50)
        self.pae_head = PredictedAlignedErrorHead(d_pair, num_bins=64)
        self.distogram_head = DistogramHead(d_pair, num_bins=64)
        self.masked_msa_head = MaskedMSAHead(d_msa, num_classes=23)

        self.loss_fn = AlphaFoldLoss()

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        compute_loss: bool = False,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        num_recycles: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        if num_recycles is None:
            num_recycles = self.num_recycles

        if verbose:
            print("=" * 60)
            print("ALPHAFOLD2 FORWARD PASS")
            print("=" * 60)

        msa_emb, pair_emb = self.input_embedding(features['msa_feat'], features['pair'])

        prev_msa = None
        prev_pair = None
        prev_positions = None

        for recycle_idx in range(num_recycles + 1):
            if verbose:
                print(f"\n--- Recycle {recycle_idx}/{num_recycles} ---")

            msa_recycled, pair_recycled = self.recycling_embedder(
                msa_emb, pair_emb, prev_msa, prev_pair, prev_positions
            )

            if verbose:
                print("  Running Evoformer...")
            msa_refined, pair_refined = self.evoformer(
                msa_recycled, pair_recycled, features['msa_mask'], features['pair_mask'], verbose=verbose
            )

            single = self.msa_to_single(msa_refined[:, 0])

            if verbose:
                print("  Running Structure Module...")
            positions, frames = self.structure_module(single, pair_refined, features['seq_mask'], verbose=verbose)

            prev_msa = msa_refined.detach()
            prev_pair = pair_refined.detach()
            prev_positions = positions.detach()

        plddt_logits = self.plddt_head(single)
        plddt = PredictedLDDTHead.compute_plddt(plddt_logits)
        pae_logits = self.pae_head(pair_refined)
        pae_results = self.pae_head.compute_pae(pae_logits)
        distogram_logits = self.distogram_head(pair_refined)
        masked_msa_logits = self.masked_msa_head(msa_refined)

        outputs = {
            'positions': positions,
            'frames': frames,
            'plddt': plddt,
            'plddt_logits': plddt_logits,
            'pae': pae_results['predicted_aligned_error'],
            'pae_logits': pae_logits,
            'distogram_logits': distogram_logits,
            'masked_msa_logits': masked_msa_logits,
            'msa': msa_refined,
            'pair': pair_refined,
            'single': single
        }

        if verbose:
            print(f"\nOutputs:")
            print(f"  Positions: {positions.shape}")
            print(f"  pLDDT: [{plddt.min():.1f}, {plddt.max():.1f}]")

        return outputs

    def predict_structure(self, msa_feat, pair_feat, msa_mask, pair_mask, seq_mask, num_recycles=None):
        features = {'msa_feat': msa_feat, 'pair': pair_feat, 'msa_mask': msa_mask,
                    'pair_mask': pair_mask, 'seq_mask': seq_mask}
        with torch.no_grad():
            outputs = self(features, num_recycles=num_recycles)
        return {'positions': outputs['positions'], 'plddt': outputs['plddt'], 'pae': outputs['pae']}


if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE ALPHAFOLD2 MODEL TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    batch, N_seq, L = 1, 5, 32

    model = AlphaFold2(
        d_msa=256, d_pair=128, d_single=384,
        num_evoformer_blocks=2,
        num_structure_layers=2,
        num_recycles=1
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    features = {
        'msa_feat': torch.randn(batch, N_seq, L, 49, device=device),
        'pair': torch.randn(batch, L, L, 506, device=device),
        'msa_mask': torch.ones(batch, N_seq, L, device=device),
        'pair_mask': torch.ones(batch, L, L, device=device),
        'seq_mask': torch.ones(batch, L, device=device)
    }

    outputs = model(features, verbose=True)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Positions: {outputs['positions'].shape}")
    print(f"pLDDT: {outputs['plddt'].shape}, range [{outputs['plddt'].min():.1f}, {outputs['plddt'].max():.1f}]")
    print(f"PAE: {outputs['pae'].shape}")

    loss = outputs['positions'].sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")
