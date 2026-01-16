import torch
from config import Config
from Transformer import Transformer
from dataset import create_data_loader
from tokenizer import Tokenizer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(Config())
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)
    
    data_loader = create_data_loader("./data.txt",Config())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for inputs,targets in data_loader:

            inputs,targets = inputs.to(device),targets.to(device)

            out = model(inputs)
    
            loss = loss_fn(out.view(-1, Config().vocab_size), targets.view(-1))
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print("Epoch", epoch, "Loss", total_loss / num_batches)



def inference():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(config)

    input_text = "Hello, how are you?"
    input_ids = tokenizer.encode(input_text).squeeze(0).to(device)
    
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    max_new_tokens = 100
    
    print(f"Prompt: {input_text}\n")
    print("Generating: ", end='', flush=True)  # Start generation line
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            out = model(input_ids)
            next_token = out[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            
            # Limit context window
            if input_ids.shape[1] > config.max_position_embeddings:
                input_ids = input_ids[:, -config.max_position_embeddings:]
            
            # Decode only the new token
            new_token_text = tokenizer.decode([next_token.cpu().tolist()])
            
            # Print new token on same line
            print(new_token_text, end='', flush=True)
    
    print()  # New line at the end


if __name__ == "__main__":
    inference()


    


    