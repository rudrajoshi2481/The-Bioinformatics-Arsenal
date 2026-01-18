import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from Attention import MultiHeadAttention


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.attention_layer_norm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.fc1 = nn.Linear(config.embed_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_size)
        self.ffn_layer_norm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, hidden_states, attention_mask=None, need_weights=False):
        # Self-attention with residual connection
        residual = hidden_states
        attn_output, attn_weights = self.attention(
            query=hidden_states,
            key=hidden_states,     
            value=hidden_states,    
            key_padding_mask=attention_mask,
            need_weights=need_weights
        )
        hidden_states = residual + self.dropout(attn_output)
        hidden_states = self.attention_layer_norm(hidden_states)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states)
        
        return hidden_states, attn_weights


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        all_hidden_states = [] if output_attentions else None
        all_attentions = [] if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_attentions:
                all_hidden_states.append(hidden_states)
            
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, need_weights=False)
                    return custom_forward
                
                hidden_states, attn_weights = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask,
                    need_weights=output_attentions
                )
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }
