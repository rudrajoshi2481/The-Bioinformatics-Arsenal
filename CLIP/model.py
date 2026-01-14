import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from tqdm import tqdm
import math

from config import config
from TextTransformer import TextTransformer
from VisionTransformer import VisionTransformer


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = TextTransformer(config['text'])
        self.vision_encoder = VisionTransformer(config['vision'])
        
        self.text_projection = nn.Linear(
            config['text']['embedding_dim'],
            config['text']['output_dim'],
            bias=False
        )
        
        self.vision_projection = nn.Linear(
            config['vision']['embedding_dim'],
            config['vision']['output_dim'],
            bias=False
        )
        
        self.temperature = nn.Parameter(
            torch.ones([]) * config['training']['temperature']
        )
        
    def encode_text(self, text):
        x = self.text_encoder(text)
        
        eos_indices = (text == 49407).int().argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
        
        x = self.text_projection(x)
        x = F.normalize(x, dim=-1)
        return x
    
    def encode_image(self, images):
        x = self.vision_encoder(images)
        x = self.vision_projection(x)
        x = F.normalize(x, dim=-1)
        return x
    
    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        return image_features, text_features


def clip_loss(image_features, text_features, temperature):
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2, logits


def compute_accuracy(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    i2t_preds = logits.argmax(dim=1)
    i2t_acc = (i2t_preds == labels).float().mean()
    
    t2i_preds = logits.T.argmax(dim=1)
    t2i_acc = (t2i_preds == labels).float().mean()
    
    return i2t_acc.item(), t2i_acc.item()


def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


model = CLIPModel(config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")
print(model)
