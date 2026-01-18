#!/usr/bin/env python3
"""
CPU Test Script for ESM2 Training Pipeline
Tests the full training loop with synthetic data on CPU (no DDP)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import tempfile
import os

from tokenizer import ESMTokenizer
from model import ESMForMaskedLM
from config import MLMConfig
from dataset import MLMDataset


def generate_synthetic_data(num_samples=100, min_len=20, max_len=100):
    """Generate random protein sequences"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    data = []
    for _ in range(num_samples):
        length = torch.randint(min_len, max_len, (1,)).item()
        seq = ''.join([amino_acids[torch.randint(0, len(amino_acids), (1,)).item()] 
                       for _ in range(length)])
        data.append({"sequence": seq})
    return data


def test_dataset():
    """Test the MLMDataset with synthetic data"""
    print("=" * 60)
    print("Test 1: MLMDataset with Synthetic Data")
    print("=" * 60)
    
    tokenizer = ESMTokenizer()
    
    # Create temp JSON file with synthetic data
    data = generate_synthetic_data(num_samples=50)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_file = f.name
    
    try:
        dataset = MLMDataset(
            json_file=temp_file,
            tokenizer=tokenizer,
            max_length=128,
            mlm_prob=0.15
        )
        
        # Test single item
        item = dataset[0]
        print(f"✓ Dataset created with {len(dataset)} samples")
        print(f"  input_ids shape: {item['input_ids'].shape}")
        print(f"  attention_mask shape: {item['attention_mask'].shape}")
        print(f"  labels shape: {item['labels'].shape}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"✓ DataLoader working")
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        
        return temp_file, tokenizer
        
    except Exception as e:
        os.unlink(temp_file)
        raise e


def test_model_forward():
    """Test model forward pass"""
    print("\n" + "=" * 60)
    print("Test 2: Model Forward Pass")
    print("=" * 60)
    
    config = MLMConfig()
    model = ESMForMaskedLM(config)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(2, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :10] = -100  # Ignore first 10 positions
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    print(f"✓ Forward pass successful")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    return model, config


def test_training_loop(temp_file, tokenizer):
    """Test a few training steps"""
    print("\n" + "=" * 60)
    print("Test 3: Training Loop (5 steps)")
    print("=" * 60)
    
    config = MLMConfig()
    model = ESMForMaskedLM(config)
    model.train()
    
    # Create dataset and loader
    dataset = MLMDataset(
        json_file=temp_file,
        tokenizer=tokenizer,
        max_length=128,
        mlm_prob=0.15
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    losses = []
    for step, batch in enumerate(loader):
        if step >= 5:
            break
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
    
    print(f"✓ Training loop completed")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    return model


def test_gradient_flow():
    """Test that gradients flow correctly"""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    config = MLMConfig()
    model = ESMForMaskedLM(config)
    model.train()
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(2, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs['loss']
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            norm = param.grad.norm().item()
            if 'esm.embeddings' in name:
                grad_norms['embeddings'] = norm
            elif 'esm.transformer.layers.0' in name and 'attention' in name:
                grad_norms['attention_layer_0'] = norm
            elif 'lm_head' in name:
                grad_norms['lm_head'] = norm
    
    print(f"✓ Gradients computed: {has_grad}")
    for key, norm in grad_norms.items():
        print(f"  {key} grad norm: {norm:.6f}")


def main():
    print("=" * 60)
    print("ESM2 Training Pipeline - CPU Test")
    print("=" * 60)
    print("Testing with synthetic data on CPU...\n")
    
    try:
        # Test 1: Dataset
        temp_file, tokenizer = test_dataset()
        
        # Test 2: Model forward
        test_model_forward()
        
        # Test 3: Training loop
        test_training_loop(temp_file, tokenizer)
        
        # Test 4: Gradient flow
        test_gradient_flow()
        
        # Cleanup
        os.unlink(temp_file)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe training pipeline is ready for GPU cluster deployment.")
        print("Make sure to sync the fixed dataset.py to your cluster!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
