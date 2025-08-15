from transformers import AutoTokenizer

class BPETokenizer:
    def __init__(self, vocab_path):
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_path)

    def encode(self, text):
        # Return a list of token IDs just like the original
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        # Return decoded string from token IDs
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
