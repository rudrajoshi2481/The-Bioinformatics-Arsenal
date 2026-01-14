import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionAttention(nn.Module):
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
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)

        B, N, E = x.shape
        q = self.query(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.attn_out_proj(out)
        out = self.attn_dropout(out)
        
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.patch_size = config['patch_size']
        self.embedding_dim = config['embedding_dim']
        self.image_size = config['image_size']
        self.n_channels = config.get('n_channels', 3)
        
        num_patches = (self.image_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(self.n_channels, self.embedding_dim, 
                                      kernel_size=self.patch_size, 
                                      stride=self.patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.embedding_dim) * 0.02)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.layers = nn.ModuleList([
            VisionAttention(config) for _ in range(config['n_layers'])
        ])
        
        self.norm = nn.LayerNorm(self.embedding_dim)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x[:, 0]


# Test code at end of VisionTransformer.py
if __name__ == '__main__':
    config = {
        'image_size': 224,
        'patch_size': 32,
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 4,
        'dropout': 0.1,
        'n_channels': 3,
    }
    
    model = VisionTransformer(config)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: torch.Size([2, 256])")
    
    if out.shape == torch.Size([2, 256]):
        print("✅ Shape correct!")
    else:
        print("❌ Shape wrong!")

