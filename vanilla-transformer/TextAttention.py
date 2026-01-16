import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from encoding import ROPEEncoding


class TextAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers, config):
        super(TextAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = ROPEEncoding(max_position_embeddings=config.max_position_embeddings, hidden_size=d_model // nhead)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x, mask=True):
        
        residual = x
        x = self.pre_norm(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        def reshape(x):
            batch_size, seq_len, d_model = x.size()
            return x.view(batch_size, seq_len, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        
        q = reshape(q)
        k = reshape(k)
        v = reshape(v)
        
        q, k = self.rope(q, k)
        
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.nhead), dim=-1)

        if mask:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).expand(x.size(0), seq_len, seq_len).to(x.device)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(residual.size())
        out = self.out_proj(out)
        out = self.dropout(out)
        
        x = residual + out  
        
        
        residual = x  
        x = self.post_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return residual + x  

