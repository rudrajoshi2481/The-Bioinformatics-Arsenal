
from tokenizer import ESMTokenizer
from mlm_collator import DataCollatorForMLM
import torch

if __name__ == "__main__":
    print("="*60)
    print("Testing MLM Pipeline")
    print("="*60)
    
    # 1. Test Tokenizer
    print("\n1. Testing Tokenizer")
    print("-" * 60)
    tokenizer = ESMTokenizer()
    
    test_sequence = "MKTAYIAKQRQISFVK"
    encoded = tokenizer.encode(test_sequence, add_special_tokens=True)
    
    print(f"Original sequence: {test_sequence}")
    print(f"Token IDs: {encoded['input_ids']}")
    print(f"Attention mask: {encoded['attention_mask']}")
    
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded sequence: {decoded}")
    assert decoded == test_sequence, "Encoding/decoding mismatch!"
    print("✓ Tokenizer working\n")
    
    # 2. Test Batch Encoding
    print("2. Testing Batch Encoding")
    print("-" * 60)
    sequences = ["MKTAY", "AKQRQISFVKSHFSRQLE", "DLGR"]
    batch = tokenizer.batch_encode(sequences, max_length=25, padding=True)
    
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print("✓ Batch encoding working\n")
    
    # 3. Test MLM Collator
    print("3. Testing MLM Masking")
    print("-" * 60)
    collator = DataCollatorForMLM(tokenizer, mlm_probability=0.15)
    
    examples = [
        tokenizer.encode("MKTAYIAKQRQISFVK", add_special_tokens=True, max_length=25, padding=True),
        tokenizer.encode("AKQRQISFVKSHFSRQLE", add_special_tokens=True, max_length=25, padding=True),
    ]
    
    batch = collator(examples)
    
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"\nExample masking:")
    print(f"Original: {examples[0]['input_ids'][:20]}")
    print(f"Masked:   {batch['input_ids'][0, :20]}")
    print(f"Labels:   {batch['labels'][0, :20]}")
    
    # Count masked positions
    masked_count = (batch['labels'][0] != -100).sum().item()
    total_tokens = (examples[0]['attention_mask'] == 1).sum().item()
    print(f"\nMasked {masked_count}/{total_tokens} tokens ({masked_count/total_tokens*100:.1f}%)")
    print("✓ MLM masking working\n")
    
    # 4. Test with Model
    print("4. Testing Full Pipeline with Model")
    print("-" * 60)
    from config import MLMConfig
    from model import ESMForMaskedLM
    
    config = MLMConfig()
    model = ESMForMaskedLM(config)
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Get predictions for masked positions
    masked_positions = (batch['labels'][0] != -100).nonzero(as_tuple=True)[0]
    if len(masked_positions) > 0:
        pos = masked_positions[0].item()
        true_token = batch['labels'][0, pos].item()
        predicted_token = outputs['logits'][0, pos].argmax().item()
        
        print(f"\nExample prediction at position {pos}:")
        print(f"  True token: {tokenizer.id_to_token[true_token]}")
        print(f"  Predicted token: {tokenizer.id_to_token[predicted_token]}")
    
    print("\n✓ Full pipeline working\n")
    
    print("="*60)
    print("ALL MLM COMPONENTS TESTED ✓")
    print("="*60)