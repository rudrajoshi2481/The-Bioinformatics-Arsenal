import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ROPEEncoding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, base=10000):
        super(ROPEEncoding, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
        
        self._set_cos_sin_cache(max_position_embeddings)
        
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        
        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[-2]
            
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


# Example usage
# if __name__ == "__main__":
#     batch_size = 2
#     num_heads = 8
#     seq_len = 128
#     head_dim = 64
    
#     # Initialize RoPE
#     rope = ROPEEncoding(max_position_embeddings=512, hidden_size=head_dim)
    
#     # Create sample Q and K tensors
#     q = torch.randn(batch_size, num_heads, seq_len, head_dim)
#     k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
#     # Apply RoPE
#     q_rotated, k_rotated = rope(q, k)
    
#     print(f"Original Q shape: {q.shape}")
#     print(f"Rotated Q shape: {q_rotated.shape}")
#     print(f"Original K shape: {k.shape}")
#     print(f"Rotated K shape: {k_rotated.shape}")
