import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from tqdm import tqdm
from model import clip_loss, compute_accuracy


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]   
        )
    ])



def print_model_info(model):
    """Print model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Statistics:")
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")


def train_step(model, images, texts, optimizer, device):
    """Single training step."""
    images = images.to(device)
    texts = texts.to(device)
    
    optimizer.zero_grad()
    
    if hasattr(model, 'module'):
        image_features = model.module.encode_image(images)
        text_features = model.module.encode_text(texts)
        temperature = model.module.temperature
    else:
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        temperature = model.temperature
    
    loss, logits = clip_loss(image_features, text_features, temperature)
    
    loss.backward()
    optimizer.step()
    
    i2t_acc, t2i_acc = compute_accuracy(logits)
    
    return loss.item(), i2t_acc, t2i_acc, logits


def validate_model(model, dataloader, device):
    """Validate model with proper distributed aggregation."""
    model.eval()
    total_loss = 0
    total_i2t_acc = 0
    total_t2i_acc = 0
    num_batches = 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validating")
        else:
            pbar = dataloader
            
        for images, texts in pbar:
            images = images.to(device)
            texts = texts.to(device)
            
            if hasattr(model, 'module'):
                image_features = model.module.encode_image(images)
                text_features = model.module.encode_text(texts)
                temperature = model.module.temperature
            else:
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                temperature = model.temperature
            
            loss, logits = clip_loss(image_features, text_features, temperature)
            i2t_acc, t2i_acc = compute_accuracy(logits)
            
            total_loss += loss.item()
            total_i2t_acc += i2t_acc
            total_t2i_acc += t2i_acc
            num_batches += 1
    
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, total_i2t_acc, total_t2i_acc, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_i2t_acc, total_t2i_acc, num_batches = metrics.tolist()
    
    avg_loss = total_loss / num_batches
    avg_i2t = total_i2t_acc / num_batches
    avg_t2i = total_t2i_acc / num_batches
    
    return avg_loss, avg_i2t, avg_t2i
