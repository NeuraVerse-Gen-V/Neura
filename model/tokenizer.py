import json
import ast

class BPETokenizer():

    def __init__(self,vocab_path):
        self.vocab_path=vocab_path
    def encode(self, text):
        #split text into words then words into pairs of tokens and then encode them using the pre-trained BPE tokenizer
        with open(self.vocab_path, "r" ,encoding="utf-8") as f:
            vocab = json.load(f)
        
        # Convert text to byte-level token IDs
        token_ids = [a if a<256 else vocab["<unk>"] for a in text.encode("utf-8")]

        # Keep merging pairs while possible
        while True:
            merged = False
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i+1])
                pair_str = str(pair)

                if pair_str in vocab:
                    token_ids[i] = vocab[pair_str]
                    del token_ids[i+1]
                    merged = True
                else:
                    i += 1

            if not merged:
                break

        return token_ids

    def decode(self, token_ids):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Reverse the vocab: token_id -> token_str (which may be a tuple string)
        reverse_vocab = {int(v): k for k, v in vocab.items()}

        def resolve_token(token):
            result = []

            if isinstance(token, list):
                for t in token:
                    result.extend(resolve_token(t))

            elif isinstance(token, int):
                if token not in reverse_vocab:
                    return []

                key = reverse_vocab[token]
                try:
                    val = ast.literal_eval(key)
                    if isinstance(val, tuple):
                        for sub in val:
                            result.extend(resolve_token(sub))
                    else:
                        result.append(val)
                except:
                    result.append(key)

            elif isinstance(token, str):
                result.append(token)

            return result

        decoded = ""
        resolved_tokens=resolve_token(token_ids)
        for token in resolved_tokens:
            if isinstance(token,str):
                decoded+=token
            else:
                decoded+=str(token)
        
        return decoded
