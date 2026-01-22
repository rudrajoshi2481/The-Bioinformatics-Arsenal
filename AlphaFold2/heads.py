import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PredictedLDDTHead(nn.Module):
    def __init__(self, d_single: int, num_bins: int = 50, d_hidden: int = 128):
        super().__init__()
        self.num_bins = num_bins
        self.layer_norm = nn.LayerNorm(d_single)
        self.linear1 = nn.Linear(d_single, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.linear3 = nn.Linear(d_hidden, num_bins)

    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(single_repr)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

    @staticmethod
    def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bin_centers = torch.linspace(0.5 * bin_width, 1.0 - 0.5 * bin_width, num_bins, device=logits.device)
        probs = F.softmax(logits, dim=-1)
        plddt = torch.sum(probs * bin_centers, dim=-1) * 100
        return plddt


class PredictedAlignedErrorHead(nn.Module):
    def __init__(self, d_pair: int, num_bins: int = 64, max_error: float = 31.0, d_hidden: int = 128):
        super().__init__()
        self.num_bins = num_bins
        self.max_error = max_error
        self.layer_norm = nn.LayerNorm(d_pair)
        self.linear1 = nn.Linear(d_pair, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.linear3 = nn.Linear(d_hidden, num_bins)
        breaks = torch.linspace(0, max_error, num_bins)
        self.register_buffer('breaks', breaks)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(pair_repr)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

    def compute_pae(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        step = self.breaks[1] - self.breaks[0]
        bin_centers = self.breaks + step / 2
        pae = torch.sum(probs * bin_centers, dim=-1)
        return {
            'aligned_confidence_probs': probs,
            'predicted_aligned_error': pae,
            'max_predicted_aligned_error': bin_centers[-1]
        }


class DistogramHead(nn.Module):
    def __init__(self, d_pair: int, num_bins: int = 64, first_break: float = 2.3125, last_break: float = 21.6875):
        super().__init__()
        self.num_bins = num_bins
        self.layer_norm = nn.LayerNorm(d_pair)
        self.linear = nn.Linear(d_pair, num_bins)
        breaks = torch.linspace(first_break, last_break, num_bins - 1)
        self.register_buffer('breaks', breaks)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(pair_repr)
        logits = self.linear(x)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits


class MaskedMSAHead(nn.Module):
    def __init__(self, d_msa: int, num_classes: int = 23):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_msa)
        self.linear = nn.Linear(d_msa, num_classes)

    def forward(self, msa_repr: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(msa_repr)
        logits = self.linear(x)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    def __init__(self, d_single: int, num_atoms: int = 37):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_single)
        self.linear = nn.Linear(d_single, num_atoms)

    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(single_repr)
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTION HEADS TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    batch, N_seq, L = 2, 10, 64
    d_msa, d_pair, d_single = 256, 128, 384

    msa_repr = torch.randn(batch, N_seq, L, d_msa, device=device)
    pair_repr = torch.randn(batch, L, L, d_pair, device=device)
    single_repr = torch.randn(batch, L, d_single, device=device)

    plddt_head = PredictedLDDTHead(d_single).to(device)
    logits = plddt_head(single_repr)
    plddt = PredictedLDDTHead.compute_plddt(logits)
    print(f"pLDDT: {logits.shape} -> scores {plddt.shape}, range [{plddt.min():.1f}, {plddt.max():.1f}]")

    pae_head = PredictedAlignedErrorHead(d_pair).to(device)
    logits = pae_head(pair_repr)
    pae = pae_head.compute_pae(logits)
    print(f"PAE: {logits.shape} -> error {pae['predicted_aligned_error'].shape}")

    dist_head = DistogramHead(d_pair).to(device)
    logits = dist_head(pair_repr)
    print(f"Distogram: {logits.shape}")

    msa_head = MaskedMSAHead(d_msa).to(device)
    logits = msa_head(msa_repr)
    print(f"Masked MSA: {logits.shape}")

    res_head = ExperimentallyResolvedHead(d_single).to(device)
    logits = res_head(single_repr)
    print(f"Experimentally Resolved: {logits.shape}")

    loss = logits.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")
