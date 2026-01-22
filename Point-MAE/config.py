class Config:
    """Point-MAE Configuration for pretraining and finetuning."""
    
    # ============== Architecture ==============
    encoder_channels = 384
    embed_size = 384
    heads = 6
    attn_dropout = 0.0
    projection_dropout = 0.0
    
    # ============== Transformer ==============
    depth = 12
    decoder_depth = 4
    decoder_num_heads = 6
    mlp_ratio = 4
    drop_path_rate = 0.1
    
    # ============== Point Cloud ==============
    input_channels = 6  # 6 for xyz+normals, 3 for xyz only (normals preserved for diffusion)
    num_group = 64
    group_size = 32
    npoints = 8192
    
    # ============== MAE ==============
    mask_ratio = 0.6
    mask_type = 'rand'  # 'rand' or 'block'
    
    # ============== Loss ==============
    loss = 'cdl2'  # 'cdl1' or 'cdl2'
    
    # ============== Dataset ==============
    dataset_name = 'ModelNet'
    data_path = '/app/tmp/Point-MAE/data'
    num_category = 40
    use_normals = True  # Keep normals for future diffusion head
    num_workers = 4
    
    # ============== Training ==============
    batch_size = 32  # per GPU
    max_epoch = 300
    warmup_epochs = 10
    lr = 0.001
    min_lr = 1e-6
    weight_decay = 0.05
    grad_norm_clip = 10
    
    # ============== DDP Training ==============
    use_amp = True  # Mixed precision training
    gradient_accumulation_steps = 1
    
    # ============== Output ==============
    output_dir = './output'
    checkpoint_dir = './output/checkpoints'
    log_dir = './output/logs'
    plot_dir = './output/plots'
    vis_dir = './output/visualizations'
    
    # ============== Logging ==============
    log_every = 10  # Log every N steps
    save_every_n_steps = 500
    val_every_n_epochs = 1
    log_to_file = True  # Always generate log files
    log_metrics = ['loss', 'lr', 'grad_norm', 'throughput', 'gpu_memory']
    
    # ============== Visualization ==============
    vis_every_n_epochs = 5
    num_vis_samples = 4
    
    def get(self, key, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() 
                 if not k.startswith('_') and not callable(v)}
        return f"Config({attrs})"


if __name__ == "__main__":
    config = Config()
    
    print("=" * 60)
    print("Point-MAE Configuration")
    print("=" * 60)
    
    print("\n[Dataset]")
    print(f"  Name: {config.dataset_name} ({config.num_category} classes)")
    print(f"  Path: {config.data_path}")
    print(f"  Points: {config.npoints}")
    print(f"  Channels: {config.input_channels} ({'xyz+normals' if config.use_normals else 'xyz'})")
    
    print("\n[Architecture]")
    print(f"  Embed size: {config.embed_size}")
    print(f"  Heads: {config.heads} (head_dim={config.embed_size//config.heads})")
    print(f"  Encoder depth: {config.depth}")
    print(f"  Decoder depth: {config.decoder_depth}")
    print(f"  MLP ratio: {config.mlp_ratio}")
    
    print("\n[Grouping]")
    print(f"  Groups: {config.num_group}")
    print(f"  Points/group: {config.group_size}")
    print(f"  Total grouped: {config.num_group * config.group_size}")
    
    print("\n[MAE]")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Mask type: {config.mask_type}")
    print(f"  Masked groups: {int(config.num_group * config.mask_ratio)}")
    print(f"  Visible groups: {int(config.num_group * (1 - config.mask_ratio))}")
    
    print("\n[Training]")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Epochs: {config.max_epoch}")
    print(f"  Warmup epochs: {config.warmup_epochs}")
    print(f"  Mixed precision: {config.use_amp}")
    
    print("\n" + "=" * 60)
