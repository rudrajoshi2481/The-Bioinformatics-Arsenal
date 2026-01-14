import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import open_clip  # pip install open-clip-torch


class CLIPDataset(Dataset):
    """
    CLIP Dataset with proper tokenization

    Args:
        json_file: Path to JSON file with image_path and caption
        transform: Image transformations
        tokenizer: Optional - if None, will use ViT-B-32 tokenizer
    """
    def __init__(self, json_file, transform=None, tokenizer=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

        
        if tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_tokens = self.tokenizer(item['caption']).squeeze(0)

        return image, text_tokens
