import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from config import config
from model import CLIPModel, clip_loss, compute_accuracy, get_lr
from dataset import CLIPDataset
from utils import train_step, validate_model, get_transforms, print_model_info
from torchvision import transforms


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os



def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank



def cleanup_distributed():
    dist.destroy_process_group()



def train_epoch(model, dataloader, optimizer, epoch, config, device, rank):
    model.train()
    total_loss = 0
    total_i2t_acc = 0
    total_t2i_acc = 0
    
    lr = get_lr(
        epoch,
        config['training']['warmup_epochs'],
        config['training']['epochs'],
        config['training']['learning_rate']
    )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader
    
    for images, texts in pbar:
        loss, i2t_acc, t2i_acc, logits = train_step(
            model, images, texts, optimizer, device
        )
        
        total_loss += loss
        total_i2t_acc += i2t_acc
        total_t2i_acc += t2i_acc
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'i2t': f'{i2t_acc*100:.1f}%',
                't2i': f'{t2i_acc*100:.1f}%',
                'lr': f'{lr:.2e}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_i2t = total_i2t_acc / len(dataloader)
    avg_t2i = total_t2i_acc / len(dataloader)
    
    return avg_loss, avg_i2t, avg_t2i



def main():
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    save_dir = '/data/joshi/utils/junks'
    os.makedirs(save_dir, exist_ok=True)

    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Device: {device}")
        print(f"Save directory: {save_dir}")
    
    transform = get_transforms()


    # Print the transform to see what normalization is being used
    print("\n" + "="*80)
    print("CHECKING TRANSFORMS")
    print("="*80)
    for t in transform.transforms:
        print(f"  {t}")
        if isinstance(t, transforms.Normalize):
            print(f"    Mean: {t.mean}")
            print(f"    Std: {t.std}")
    print("="*80 + "\n")
    
    train_dataset = CLIPDataset(
        '/data/joshi/utils/junks/imagenet100_clip/train.json',
        transform=transform
    )
    
    val_dataset = CLIPDataset(
        '/data/joshi/utils/junks/imagenet100_clip/test.json',
        transform=transform
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        print(f"Batch size per GPU: {config['training']['batch_size']}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
    
    model = CLIPModel(config).to(device)


    if rank == 0:
        print_model_info(model)


    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas'],
        eps=config['training']['eps']
    )
    
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    if rank == 0:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
    
    for epoch in range(config['training']['epochs']):
        train_sampler.set_epoch(epoch)


        train_loss, train_i2t, train_t2i = train_epoch(
            model, train_loader, optimizer, epoch, config, device, rank
        )
        
        val_loss, val_i2t, val_t2i = validate_model(model, val_loader, device)
        
        train_avg = (train_i2t + train_t2i) / 2
        val_avg = (val_i2t + val_t2i) / 2
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train: Loss={train_loss:.4f} Acc={train_avg*100:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f} Acc={val_avg*100:.2f}%")
        
        if val_avg > best_val_acc:
            best_val_acc = val_avg
            patience_counter = 0
            if rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'val_acc': val_avg,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  ✅ Best: {val_avg*100:.2f}%")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 and rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_avg,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        if rank == 0:
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
        
        should_stop = torch.tensor(patience_counter >= patience, device=device)
        dist.broadcast(should_stop, src=0)
        if should_stop.item():
            break
        
        if rank == 0:
            print("="*80)
    
    if rank == 0:
        print(f"\nTraining complete! Best: {best_val_acc*100:.2f}%")
        print(f"Models saved to: {save_dir}")
    
    cleanup_distributed()



if __name__ == '__main__':
    main()
