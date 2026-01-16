import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchinfo import summary

from config import Config
from TextAttention import TextAttention
from embeddings import Embeddings

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.attention = nn.ModuleList([TextAttention(config.d_model, config.nhead, config.num_layers, config) for _ in range(config.num_layers)])

        self.embeddings = Embeddings(config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, text):
        x = self.embeddings(text)
        
        for attention in self.attention:
            x = attention(x)
        
        x = self.lm_head(x)
        return x


def main():
    config = Config()
    model = Transformer(config)
    summary(
    model, 
    input_size=(8, 512), 
    dtypes=[torch.long],
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=4,
    row_settings=["var_names"]
)

if __name__ == "__main__":
    main()

