import torch
from typing import List, Dict, Union


class ESMTokenizer:
    """ESM2 tokenizer for protein sequences"""
    
    def __init__(self):
        # ESM2 vocabulary (33 tokens)
        self.standard_amino_acids = [
            'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
            'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
            'X', 'B', 'U', 'Z', 'O', '.'
        ]
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<cls>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<mask>': 32
        }
        
        # Build vocabulary
        self.vocab = self.special_tokens.copy()
        for i, aa in enumerate(self.standard_amino_acids):
            self.vocab[aa] = i + 4  # Start after special tokens
        
        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab['<pad>']
        self.cls_token_id = self.vocab['<cls>']
        self.eos_token_id = self.vocab['<eos>']
        self.mask_token_id = self.vocab['<mask>']
        self.unk_token_id = self.vocab['<unk>']
        
    def __len__(self):
        return self.vocab_size
    
    def __call__(self, sequence: str, max_length: int = None, padding: str = None,
                 truncation: bool = False, return_tensors: str = None) -> Dict[str, torch.Tensor]:
        """
        Make tokenizer callable like HuggingFace tokenizers.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length
            padding: 'max_length' to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: 'pt' for PyTorch tensors (default behavior)
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Tokenize with special tokens
        token_ids = [self.cls_token_id]
        for aa in sequence:
            token_ids.append(self.vocab.get(aa, self.unk_token_id))
        token_ids.append(self.eos_token_id)
        
        # Truncate if needed
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if needed
        if padding == 'max_length' and max_length is not None:
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        # Return as tensors (add batch dimension for compatibility)
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        }
        
    def encode(self, sequence: str, add_special_tokens: bool = True, 
               max_length: int = None, padding: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode protein sequence to token IDs
        
        Args:
            sequence: Protein sequence string (e.g., "MKTAY...")
            add_special_tokens: Add <cls> and <eos>
            max_length: Maximum sequence length
            padding: Pad to max_length
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Tokenize
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.cls_token_id)
        
        for aa in sequence:
            token_ids.append(self.vocab.get(aa, self.unk_token_id))
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Truncate if needed
        if max_length is not None:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if needed
        if padding and max_length is not None:
            pad_length = max_length - len(token_ids)
            token_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def batch_encode(self, sequences: List[str], add_special_tokens: bool = True,
                     max_length: int = None, padding: bool = True) -> Dict[str, torch.Tensor]:
        """Batch encode multiple sequences"""
        if max_length is None and padding:
            # Find max length in batch
            max_length = max(len(seq) for seq in sequences)
            if add_special_tokens:
                max_length += 2  # For <cls> and <eos>
        
        batch = [self.encode(seq, add_special_tokens, max_length, padding) 
                 for seq in sequences]
        
        return {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch])
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to sequence"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, '<unk>')
            if skip_special_tokens and token in ['<pad>', '<cls>', '<eos>', '<mask>']:
                continue
            tokens.append(token)
        
        return ''.join(tokens)
