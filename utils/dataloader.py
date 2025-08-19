import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from model.tokenizer import BPETokenizer
from tqdm import tqdm
from functools import lru_cache
from utils.config import src_pad_idx as pad_token_id, trg_sos_idx, eos_token


# -----------------------------
# Load CSV
# -----------------------------
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading data from {file_path}: {e}")
        return None


# -----------------------------
# Cached Tokenizer
# -----------------------------
_tokenizer = None
def get_tokenizer(path="gpt2"):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BPETokenizer(path)
    return _tokenizer


# -----------------------------
# Encode with cache
# -----------------------------
@lru_cache(maxsize=50000)
def encode_text(text, tokenizer_path="gpt2"):
    tokenizer = get_tokenizer(tokenizer_path)
    return tokenizer.encode(text)


# -----------------------------
# Convert input/output → padded tensors
# -----------------------------
def tensorize(input_labels, output_labels, tokenizer_path="gpt2"):
    inp_tensors, out_tensors = [], []

    for x in tqdm(input_labels, desc="Encoding input labels"):
        inp_tensors.append(torch.tensor(encode_text(x, tokenizer_path), dtype=torch.long))

    for x in tqdm(output_labels, desc="Encoding output labels"):
        tokens = encode_text(x, tokenizer_path)
        tokens = [trg_sos_idx] + tokens + [eos_token]   # <sos> ... tokens ... <eos>
        out_tensors.append(torch.tensor(tokens, dtype=torch.long))

    inp_padded = pad_sequence(inp_tensors, batch_first=True, padding_value=pad_token_id)
    out_padded = pad_sequence(out_tensors, batch_first=True, padding_value=pad_token_id)
    return inp_padded, out_padded
