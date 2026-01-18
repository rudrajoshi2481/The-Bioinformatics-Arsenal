import torch
import random
from typing import List, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from tokenizer import ESMTokenizer


@dataclass
class DataCollatorForMLM:
    """
    Data collator for Masked Language Modeling
    Randomly masks tokens for MLM training
    """
    tokenizer: "ESMTokenizer"
    mlm_probability: float = 0.15
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Apply MLM masking to a batch
        
        Args:
            examples: List of dicts with 'input_ids' and 'attention_mask'
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Stack batch
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
        
        # Clone for labels
        labels = input_ids.clone()
        
        # Create probability matrix
        batch_size, seq_len = input_ids.shape
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask |= (input_ids == self.tokenizer.pad_token_id)
        special_tokens_mask |= (input_ids == self.tokenizer.cls_token_id)
        special_tokens_mask |= (input_ids == self.tokenizer.eos_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Don't mask padding tokens
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        # Sample positions to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~masked_indices] = -100
        
        # 80% of the time: replace with <mask>
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time: replace with random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(
            4, self.tokenizer.vocab_size - 1,  # Exclude special tokens
            input_ids.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]
        
        # 10% of the time: keep original (already done, no replacement)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }