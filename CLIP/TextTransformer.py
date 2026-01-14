import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding_dim = config['embedding_dim']
        self.dropout = config['dropout']
        self.n_heads = config['n_heads']
        self.head_dim = self.embedding_dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attn_out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
            nn.Dropout(self.dropout),
        )
        
        self.attn_dropout = nn.Dropout(self.dropout)
    
    def forward(self, x, attn_mask=None):
        residual = x
        x = self.norm1(x)

        B, S, E = x.shape
        q = self.query(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.attn_out_proj(out)
        out = self.attn_dropout(out)
        
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class TextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.positional_embedding = nn.Parameter(torch.randn(config['max_seq_len'], config['embedding_dim']) * 0.02)
        
        self.layers = nn.ModuleList([
            TextAttention(config) for _ in range(config['n_layers'])
        ])
        self.norm = nn.LayerNorm(config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:x.size(1), :]
        x = self.dropout(x)
        
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        return x
