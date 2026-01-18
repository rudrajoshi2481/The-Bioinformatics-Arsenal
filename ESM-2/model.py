# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torchinfo import summary

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import ESMEmbeddings
from Transformer import Transformer


class ESMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embeddings = ESMEmbeddings(config)
        self.transformer = Transformer(config)
        
    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        
        # Invert mask for MultiHeadAttention (True = padding to ignore)
        # attention_mask: [batch, seq] where 1 = valid, 0 = padding
        # key_padding_mask: [batch, seq] where True = padding, False = valid
        key_padding_mask = ~attention_mask
        
        # Get embeddings [seq, batch, embed]
        hidden_states = self.embeddings(input_ids)
        
        # Pass through transformer
        # Note: hidden_states is [seq, batch, embed]
        # key_padding_mask should be [batch, seq] - already correct!
        outputs = self.transformer(
            hidden_states,
            attention_mask=key_padding_mask,
            output_attentions=output_attentions or output_hidden_states
        )
        
        # Convert back to [batch, seq, embed]
        last_hidden_state = outputs['last_hidden_state'].transpose(0, 1)
        
        return {
            'last_hidden_state': last_hidden_state,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }


class ESMForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.esm = ESMModel(config)
        
        self.lm_head = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size),
            nn.GELU(),
            nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps),
            nn.Linear(config.embed_size, config.vocab_size, bias=False)
        )
        
        self.lm_head[-1].weight = self.esm.embeddings.token_embedding.weight
        
    def forward(self, input_ids, attention_mask=None, labels=None, output_attentions=False, output_hidden_states=False):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['last_hidden_state']
        prediction_scores = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }


if __name__ == "__main__":
    from config import MLMConfig
    
    config = MLMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("Testing Complete ESM2 Model")
    print("="*60)
    print(f"Device: {device}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embed size: {config.embed_size}")
    print(f"Layers: {config.layers}")
    print(f"Heads: {config.heads}\n")
    
    model = ESMForMaskedLM(config).to(device)
    model.eval()
    
    batch_size = 4
    seq_len = 128
    
    # Create input
    input_ids = torch.randint(2, config.vocab_size, (batch_size, seq_len)).to(device)  # Start from 2 to avoid padding
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
    
    # Create labels for MLM (mask position 4 means token ID 4 was masked)
    labels = input_ids.clone()
    labels[:, :] = -100  # Ignore all positions
    labels[:, 10:15] = input_ids[:, 10:15]  # Only calculate loss on positions 10-14
    
    print("Test 1: Forward pass (eval mode)")
    print("-" * 60)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False,
            output_hidden_states=False
        )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Forward pass successful\n")
    
    print("Test 2: Backward pass (train mode)")
    print("-" * 60)
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients computed: {has_grad}")
    print("✓ Backward pass successful\n")
    
    print("Test 3: Model size")
    print("-" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (fp32)")
    print(f"Comparable to ESM2-{config.embed_size//64}M parameters\n")
    
    print("Test 4: Variable sequence lengths")
    print("-" * 60)
    model.eval()
    seq_lengths = [50, 80, 100, 120]
    max_len = max(seq_lengths)
    
    print("Test 5: Model Parameters")
    print(model)
    summary(model)

    # Create padded batch
    padded_input = torch.full((batch_size, max_len), fill_value=1, dtype=torch.long).to(device)  # 1 = padding
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
    
    for i, length in enumerate(seq_lengths):
        padded_input[i, :length] = torch.randint(2, config.vocab_size, (length,))
        padded_mask[i, :length] = 1
    
    with torch.no_grad():
        outputs = model(input_ids=padded_input, attention_mask=padded_mask)
    
    print(f"Padded input shape: {padded_input.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print("✓ Variable length handling successful\n")
    
    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
