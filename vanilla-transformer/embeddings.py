import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        # BECAUSE WE ARE USING ROPE, WE DO NOT NEED POSITION EMBEDDINGS
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        # x = x + self.position_embedding(torch.arange(x.size(1), device=x.device))
        return self.dropout(x)