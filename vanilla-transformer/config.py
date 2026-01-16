class Config:
    def __init__(self):
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6
        self.vocab_size = 50257
        self.max_position_embeddings = 512
        self.dropout = 0.1
        self.block_size = 128
        self.batch_size = 16