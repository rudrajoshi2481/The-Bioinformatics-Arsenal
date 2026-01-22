import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSARowAttention(nn.Module):
    def __init__(self, d_msa: int, d_pair: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.d_msa = d_msa
        self.d_pair = d_pair
        self.num_heads = num_heads
        self.d_head = d_msa // num_heads

        assert d_msa % num_heads == 0, "d_msa must be divisible by num_heads"

        self.layer_norm = nn.LayerNorm(d_msa)
        self.q_proj = nn.Linear(d_msa, d_msa)
        self.k_proj = nn.Linear(d_msa, d_msa)
        self.v_proj = nn.Linear(d_msa, d_msa)
        self.bias_proj = nn.Linear(d_pair, num_heads)
        self.out_proj = nn.Linear(d_msa, d_msa)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(d_msa, d_msa)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

    def forward(
        self,
        msa: torch.Tensor,
        pair_bias: torch.Tensor,
        msa_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, N_seq, L, d_msa = msa.shape

        msa_norm = self.layer_norm(msa)

        q = self.q_proj(msa_norm).view(batch, N_seq, L, self.num_heads, self.d_head)
        k = self.k_proj(msa_norm).view(batch, N_seq, L, self.num_heads, self.d_head)
        v = self.v_proj(msa_norm).view(batch, N_seq, L, self.num_heads, self.d_head)

        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        bias = self.bias_proj(pair_bias).permute(0, 3, 1, 2).unsqueeze(1)
        scores = scores + bias

        mask = msa_mask.unsqueeze(2).unsqueeze(3)
        scores = scores.masked_fill(~mask.bool(), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(2, 3).contiguous().view(batch, N_seq, L, d_msa)

        output = self.out_proj(attn_output)
        gate = torch.sigmoid(self.gate(msa_norm))
        output = output * gate

        return output


if __name__ == "__main__":
    print("=" * 70)
    print("MSA ROW ATTENTION TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    batch, N_seq, L = 2, 10, 64
    d_msa, d_pair = 256, 128

    msa = torch.randn(batch, N_seq, L, d_msa, device=device)
    pair = torch.randn(batch, L, L, d_pair, device=device)
    msa_mask = torch.ones(batch, N_seq, L, device=device)

    row_attn = MSARowAttention(d_msa, d_pair).to(device)
    output = row_attn(msa, pair, msa_mask)
    
    print(f"Input MSA: {msa.shape}")
    print(f"Input Pair: {pair.shape}")
    print(f"Output: {output.shape}")
    
    params = sum(p.numel() for p in row_attn.parameters())
    print(f"Parameters: {params:,}")
    
    loss = output.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")
