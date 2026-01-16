import torch
import torch.utils.data as data
from tokenizer import Tokenizer
from config import Config

class TextDataset(data.Dataset):
    def __init__(self, file_path,config):
        super(TextDataset, self).__init__()
        self.config = config
        self.tokenizer = Tokenizer(config)
        self.data = open(file_path, "r").read()
        self.data = self.tokenizer.encode(self.data).squeeze(0)
        
        self.block_size = config.block_size 
        self.batch_size = config.batch_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        input_ids = self.data[start_idx:end_idx]
        target_ids = self.data[start_idx + 1:end_idx + 1]
        
        return input_ids, target_ids

    def __repr__(self):
        return f"TextDataset({self.data})"
    
    def __str__(self):
        return str(self.data)
    
    def __call__(self, text):
        return self.tokenizer(text)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data


def create_data_loader(file_path,config):
    dataset = TextDataset(file_path,config)
    return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,num_workers=0)


def main(file_path="./data.txt"):
    config = Config()
    data_loader = create_data_loader(file_path,config)
    print(data_loader)
    print(next(iter(data_loader)))


if __name__ == "__main__":
    main()