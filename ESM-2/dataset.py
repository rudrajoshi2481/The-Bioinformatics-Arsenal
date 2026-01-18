
import json
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
import random
from torch.utils.data import DataLoader # type: ignore


class    MLMDataset(torch.utils.data.Dataset):
    def __init__(self, json_file,tokenizer,max_length=1024,mlm_prob=0.15):
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob

        # get special Tokens
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

        with open(json_file, 'r') as f:
            self.data = json.load(f)
            self.num_seq = len(self.data)
        
        print(f"Number of sequences: {self.num_seq}")
        print(f"Vocab size: {self.vocab_size}")

    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, idx):
        # get sequence
        sequence = self.data[idx]['sequence']
        
        # tokenize the sequnece
        encoding = self.tokenizer(sequence,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        
        # Apply MLM Masking
        input_ids, labels = self.mask_tokens(input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels   
        }

    def mask_tokens(self,input_ids,labels):
        """
        Apply BERT Style Masking
        - 15% of the tokens in each sequence are masked
        - of those 15%:
            - 80% of the masked tokens are replaced by the [MASK] token
            - 5% of the masked tokens are replaced by a random token
            - 10% of the masked tokens are left unchanged
        """
        input_ids = input_ids.clone()
        labels = labels.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(input_ids.shape, self.mlm_prob)
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask |= (input_ids == self.cls_token_id)
        special_tokens_mask |= (input_ids == self.eos_token_id)
        special_tokens_mask |= (input_ids == self.pad_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Select tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        # 80%: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10%: replace with random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        
        # 10%: keep original
        
        return input_ids, labels


class StreamingMLMDataset(Dataset):
    """
    Ultra memory-efficient: reads JSON line-by-line without loading entire file
    Best for very large datasets (10GB+)
    
    REQUIRES: JSON file with one sequence per line (JSONL format)
    """
    def __init__(
        self,
        jsonl_file,
        tokenizer,
        max_length=1024,
        mlm_probability=0.15
    ):
        print(f"Initializing streaming dataset from {jsonl_file}...")
        
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Get special token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = len(tokenizer)
        
        # Build index of file positions (fast, minimal memory)
        self.line_offsets = self._build_index()
        
        print(f"✅ Indexed {len(self.line_offsets):,} sequences")
        print(f"   Memory usage: ~{len(self.line_offsets) * 8 / 1024 / 1024:.1f} MB (just offsets)")
    
    def _build_index(self):
        """Build index of line positions in file"""
        offsets = []
        with open(self.jsonl_file, 'r') as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line.encode('utf-8'))
        return offsets
    
    def __len__(self):
        return len(self.line_offsets)
    
    def __getitem__(self, idx):
        """Read single line from file on-demand"""
        # Seek to position and read line
        with open(self.jsonl_file, 'r') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            data = json.loads(line)
        
        sequence = data['sequence']
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        # Apply masking (same as before)
        input_ids, labels = self.mask_tokens(input_ids, labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def mask_tokens(self, input_ids, labels):
        # Same implementation as MLMDataset
        input_ids = input_ids.clone()
        labels = labels.clone()
        
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask |= (input_ids == self.cls_token_id)
        special_tokens_mask |= (input_ids == self.eos_token_id)
        special_tokens_mask |= (input_ids == self.pad_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        
        return input_ids, labels


def create_mlm_dataloaders(
    train_json,
    val_json,
    tokenizer,
    batch_size=32,
    max_length=1024,
    num_workers=4,
    streaming=False  # Set True for ultra-large datasets
):
    """
    Create train and validation dataloaders for MLM
    """

    
    if streaming:
        print("Using streaming dataset (minimal memory)")
        train_dataset = StreamingMLMDataset(
            train_json, tokenizer, max_length=max_length
        )
        val_dataset = StreamingMLMDataset(
            val_json, tokenizer, max_length=max_length
        )
    else:
        print("Using standard dataset (loads into memory)")
        train_dataset = MLMDataset(
            train_json, tokenizer, max_length=max_length
        )
        val_dataset = MLMDataset(
            val_json, tokenizer, max_length=max_length
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader




