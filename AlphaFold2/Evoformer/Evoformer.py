import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from RowAttention import MSARowAttention
from ColumnAttention import MSAColumnAttention
from TriangleAttention import TriangleAttention


class Transition(nn.Module):
    def __init__(self, d_input: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_input * expansion
        self.layer_norm = nn.LayerNorm(d_input)
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class OuterProductMean(nn.Module):
    def __init__(self, d_msa: int, d_pair: int, d_hidden: int = 32):
        super().__init__()
        self.d_hidden = d_hidden

        self.layer_norm = nn.LayerNorm(d_msa)
        self.linear_left = nn.Linear(d_msa, d_hidden)
        self.linear_right = nn.Linear(d_msa, d_hidden)
        self.linear_out = nn.Linear(d_hidden, d_pair)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        msa = self.layer_norm(msa)

        left = self.linear_left(msa)
        right = self.linear_right(msa)

        mask = msa_mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        outer = torch.einsum('bnih,bnjh->bijh', left, right)

        mask_2d = msa_mask.unsqueeze(2) * msa_mask.unsqueeze(3)
        norm = mask_2d.sum(dim=1, keepdim=True)
        norm = norm.permute(0, 2, 3, 1)
        outer = outer / (norm + 1e-8)

        pair_update = self.linear_out(outer)

        return pair_update

class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair: int, outgoing: bool = True, dropout: float = 0.1):
        super().__init__()
        self.outgoing = outgoing

        self.layer_norm = nn.LayerNorm(d_pair)

        self.linear_left_proj = nn.Linear(d_pair, d_pair)
        self.linear_left_gate = nn.Linear(d_pair, d_pair)
        self.linear_right_proj = nn.Linear(d_pair, d_pair)
        self.linear_right_gate = nn.Linear(d_pair, d_pair)

        self.linear_gate = nn.Linear(d_pair, d_pair)
        self.linear_out = nn.Linear(d_pair, d_pair)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        pair = self.layer_norm(pair)

        left_proj = self.linear_left_proj(pair)
        left_gate = torch.sigmoid(self.linear_left_gate(pair))
        left = left_proj * left_gate

        right_proj = self.linear_right_proj(pair)
        right_gate = torch.sigmoid(self.linear_right_gate(pair))
        right = right_proj * right_gate

        mask = pair_mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        if self.outgoing:
            output = torch.einsum('bijd,bjkd->bikd', left, right)
        else:
            output = torch.einsum('bjid,bkjd->bikd', left, right)

        gate = torch.sigmoid(self.linear_gate(pair))
        output = output * gate
        output = self.linear_out(output)
        output = self.dropout(output)

        return output

class EvoformerBlock(nn.Module):
    def __init__(self, d_msa: int, d_pair: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.msa_row_attn = MSARowAttention(d_msa, d_pair, num_heads, dropout)
        self.msa_col_attn = MSAColumnAttention(d_msa, num_heads, dropout)
        self.msa_transition = Transition(d_msa, expansion=4, dropout=dropout)

        self.outer_product_mean = OuterProductMean(d_msa, d_pair, d_hidden=32)

        self.tri_attn_start = TriangleAttention(d_pair, num_heads=4, starting_node=True, dropout=dropout)
        self.tri_attn_end = TriangleAttention(d_pair, num_heads=4, starting_node=False, dropout=dropout)
        self.tri_mult_out = TriangleMultiplication(d_pair, outgoing=True, dropout=dropout)
        self.tri_mult_in = TriangleMultiplication(d_pair, outgoing=False, dropout=dropout)
        self.pair_transition = Transition(d_pair, expansion=4, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        msa = msa + self.dropout(self.msa_row_attn(msa, pair, msa_mask))
        msa = msa + self.dropout(self.msa_col_attn(msa, msa_mask))
        msa = msa + self.msa_transition(msa)

        pair = pair + self.outer_product_mean(msa, msa_mask)

        pair = pair + self.dropout(self.tri_attn_start(pair, pair_mask))
        pair = pair + self.dropout(self.tri_attn_end(pair, pair_mask))
        pair = pair + self.tri_mult_out(pair, pair_mask)
        pair = pair + self.tri_mult_in(pair, pair_mask)
        pair = pair + self.pair_transition(pair)

        return msa, pair

class Evoformer(nn.Module):
    def __init__(
        self,
        d_msa: int = 256,
        d_pair: int = 128,
        num_blocks: int = 48,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            EvoformerBlock(d_msa, d_pair, num_heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for i, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                msa, pair = torch.utils.checkpoint.checkpoint(
                    block, msa, pair, msa_mask, pair_mask, use_reentrant=False
                )
            else:
                msa, pair = block(msa, pair, msa_mask, pair_mask)

            if verbose and (i + 1) % 12 == 0:
                print(f"  Evoformer block {i+1}/{self.num_blocks}")

        return msa, pair

if __name__ == "__main__":
    print("=" * 70)
    print("EVOFORMER MODULE TEST")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    batch, N_seq, L = 2, 10, 64
    d_msa, d_pair = 256, 128
    num_blocks = 2

    msa = torch.randn(batch, N_seq, L, d_msa, device=device)
    pair = torch.randn(batch, L, L, d_pair, device=device)
    msa_mask = torch.ones(batch, N_seq, L, device=device)
    pair_mask = torch.ones(batch, L, L, device=device)

    print(f"Input: MSA {msa.shape}, Pair {pair.shape}")

    block = EvoformerBlock(d_msa, d_pair).to(device)
    msa_out, pair_out = block(msa, pair, msa_mask, pair_mask)
    print(f"Block output: MSA {msa_out.shape}, Pair {pair_out.shape}")

    evoformer = Evoformer(d_msa, d_pair, num_blocks=num_blocks).to(device)
    msa_final, pair_final = evoformer(msa, pair, msa_mask, pair_mask)
    print(f"Evoformer output: MSA {msa_final.shape}, Pair {pair_final.shape}")

    params = sum(p.numel() for p in evoformer.parameters())
    print(f"Parameters ({num_blocks} blocks): {params:,}")
    print(f"Estimated for 48 blocks: {params * 48 // num_blocks:,}")

    loss = msa_final.sum() + pair_final.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")