class MLMConfig:
    def __init__(self):
        
        self.train_json = '/data/joshi/utils/ESM_junk/mlm_train.json'
        self.valid_json = '/data/joshi/utils/ESM_junk/mlm_val.json'

        # Model
        self.max_seq_length = 1024
        self.vocab_size = 33
        self.mlm_probability = 0.15
        self.use_gradient_checkpointing = True
        self.embed_size = 768
        self.hidden_size = 768
        self.heads = 12
        self.layers = 12
        self.max_position_embeddings = 1024
        self.intermediate_size = 3072
        self.hidden_dropout = 0.1

        # Training
        self.num_epochs = 3
        self.batch_size = 32
        self.gradient_accumulation_steps = 4
        self.attention_dropout = 0.1
        self.embed_dropout = 0.1

        # Learning rate
        self.learning_rate= 4e-5
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.layer_norm_eps = 1e-12

        # Optimization
        self.use_amp = True
        
        # Distributed
        self.num_gpus = 8

        # Checkpoint
        self.checkpoint_dir = "/data/joshi/utils/ESM2_revised/checkpoints/mlm"
        self.save_every_n_steps = 100
        self.log_every = 100

        # workerer
        self.num_workers = 4

        