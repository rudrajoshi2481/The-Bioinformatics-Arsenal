import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ESMEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.embed_size,
            padding_idx=0  
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings + 2, 
            config.embed_size,
            padding_idx=0  
        )
        
        self.layer_norm = nn.LayerNorm(config.embed_size)
        self.dropout = nn.Dropout(getattr(config, 'embed_dropout', 0.1))
        
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)) + 2
        )
        
    def forward(self, input_ids, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len] 
        
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transpose to [seq, batch, embed] for transformer
        embeddings = embeddings.transpose(0, 1)
        
        return embeddings