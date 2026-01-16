from transformers import AutoTokenizer
from config import Config

class Tokenizer:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def __call__(self, text):
        return self.encode(text)

    def __len__(self):
        return len(self.tokenizer)
    
    def __repr__(self):
        return f"Tokenizer({self.tokenizer})"
    
    def __str__(self):
        return str(self.tokenizer)


def main():
    config = Config()
    tokenizer = Tokenizer(config)
    text = "Hello, world!"
    print(tokenizer(text))
    print(tokenizer.decode(tokenizer(text)))
    print("Vocab size:", len(tokenizer))
    print("Tokenizer:", tokenizer)
    print("Tokenizer str:", tokenizer)


if __name__ == "__main__":
    main()