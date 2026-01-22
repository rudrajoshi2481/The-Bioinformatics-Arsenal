import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSAColumnAttention(nn.Module):
    def __init__(self, d_msa: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.d_msa = d_msa
        self.num_heads = num_heads
        self.d_head = d_msa // num_heads

        self.layer_norm = nn.LayerNorm(d_msa)
        self.q_proj = nn.Linear(d_msa, d_msa)
        self.k_proj = nn.Linear(d_msa, d_msa)
        self.v_proj = nn.Linear(d_msa, d_msa)
        self.out_proj = nn.Linear(d_msa, d_msa)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(d_msa, d_msa)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        batch, N_seq, L, d_msa = msa.shape

        msa_norm = self.layer_norm(msa)

        msa_t = msa_norm.transpose(1, 2)

        q = self.q_proj(msa_t).view(batch, L, N_seq, self.num_heads, self.d_head)
        k = self.k_proj(msa_t).view(batch, L, N_seq, self.num_heads, self.d_head)
        v = self.v_proj(msa_t).view(batch, L, N_seq, self.num_heads, self.d_head)

        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = msa_mask.transpose(1, 2).unsqueeze(2).unsqueeze(3)
        scores = scores.masked_fill(~mask.bool(), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(2, 3).contiguous().view(batch, L, N_seq, d_msa)
        attn_output = attn_output.transpose(1, 2)

        output = self.out_proj(attn_output)
        gate = torch.sigmoid(self.gate(msa_norm))
        output = output * gate

        return output


if __name__ == "__main__":
    print("=" * 70)
    print("MSA COLUMN ATTENTION TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    batch, N_seq, L = 2, 10, 64
    d_msa = 256

    msa = torch.randn(batch, N_seq, L, d_msa, device=device)
    msa_mask = torch.ones(batch, N_seq, L, device=device)

    col_attn = MSAColumnAttention(d_msa).to(device)
    output = col_attn(msa, msa_mask)
    
    print(f"Input MSA: {msa.shape}")
    print(f"Output: {output.shape}")
    
    params = sum(p.numel() for p in col_attn.parameters())
    print(f"Parameters: {params:,}")
    
    loss = output.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")