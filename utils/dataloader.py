import pandas as pd
import torch
from model.tokenizer import BPETokenizer
from tqdm import tqdm
from multiprocessing import Pool,cpu_count

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    

# Helper function for pool workers
def encode_text(args):
    tokenizer_path, text = args
    tokenizer = BPETokenizer(tokenizer_path)
    return tokenizer.encode(text)

def tensorize(input_labels, output_labels):
    tokenizer_path = "model/vocab.json"
    # Prepare arguments as tuples (tokenizer_path, text)
    input_args = [(tokenizer_path, x) for x in input_labels]
    output_args = [(tokenizer_path, x) for x in output_labels]

    # Use Pool for parallel encoding
    with Pool(cpu_count()) as pool:
        inp = list(tqdm(pool.imap(encode_text, input_args), total=len(input_args), desc="Encoding input labels"))
        out = list(tqdm(pool.imap(encode_text, output_args), total=len(output_args), desc="Encoding output labels"))

    inp_tensor = torch.tensor(inp)
    out_tensor = torch.tensor(out)
    return inp_tensor, out_tensor