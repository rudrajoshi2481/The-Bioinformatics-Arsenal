import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TriangleAttention(nn.Module):
    def __init__(
        self,
        d_pair: int,
        num_heads: int = 4,
        starting_node: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_pair = d_pair
        self.num_heads = num_heads
        self.d_head = d_pair // num_heads
        self.starting_node = starting_node

        self.layer_norm = nn.LayerNorm(d_pair)
        self.q_proj = nn.Linear(d_pair, d_pair)
        self.k_proj = nn.Linear(d_pair, d_pair)
        self.v_proj = nn.Linear(d_pair, d_pair)
        self.bias_proj = nn.Linear(d_pair, num_heads)
        self.out_proj = nn.Linear(d_pair, d_pair)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(d_pair, d_pair)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        batch, L, _, d_pair = pair.shape

        pair_norm = self.layer_norm(pair)

        q = self.q_proj(pair_norm).view(batch, L, L, self.num_heads, self.d_head)
        k = self.k_proj(pair_norm).view(batch, L, L, self.num_heads, self.d_head)
        v = self.v_proj(pair_norm).view(batch, L, L, self.num_heads, self.d_head)

        if self.starting_node:
            q = q.permute(0, 1, 3, 2, 4)
            k = k.permute(0, 1, 3, 2, 4)
            v = v.permute(0, 3, 1, 2, 4).transpose(1, 2)
        else:
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 3, 1, 2, 4).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        mask = pair_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~mask.bool(), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        if self.starting_node:
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        else:
            attn_output = attn_output.permute(0, 3, 1, 2, 4)

        attn_output = attn_output.contiguous().view(batch, L, L, d_pair)

        output = self.out_proj(attn_output)
        gate = torch.sigmoid(self.gate(pair_norm))
        output = output * gate

        return output


if __name__ == "__main__":
    print("=" * 70)
    print("TRIANGLE ATTENTION TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    batch, L = 2, 64
    d_pair = 128

    pair = torch.randn(batch, L, L, d_pair, device=device)
    pair_mask = torch.ones(batch, L, L, device=device)

    tri_start = TriangleAttention(d_pair, starting_node=True).to(device)
    tri_end = TriangleAttention(d_pair, starting_node=False).to(device)
    
    out_start = tri_start(pair, pair_mask)
    out_end = tri_end(pair, pair_mask)
    
    print(f"Input Pair: {pair.shape}")
    print(f"Triangle Start Output: {out_start.shape}")
    print(f"Triangle End Output: {out_end.shape}")
    
    params = sum(p.numel() for p in tri_start.parameters())
    print(f"Parameters per module: {params:,}")
    
    loss = out_start.sum() + out_end.sum()
    loss.backward()
    print("Gradients computed successfully")
    print("PASSED")