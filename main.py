import torch

from model.transformer import Transformer
from model.tokenizer import BPETokenizer
from utils.config import *
#from vision import screen_reader,preprocess
#from voice import stt,tts

import time

tokenizer=BPETokenizer("model/vocab.json")
model=Transformer(src_pad_idx,trg_pad_idx,trg_sos_idx,enc_voc_size,dec_voc_size,d_model,n_heads,max_len,ffn_hidden,n_layers,drop_prob,device)
#model.load_state_dict(torch.load("model/transformer_weights.pth", map_location=device))
model.eval()

def generate():
    input="hi"    
    inp_tokens = torch.tensor(tokenizer.encode(input), dtype=torch.long, device=device).unsqueeze(0)
    
    out=model.generate(inp_tokens,max_len=256)
    output_text = tokenizer.decode(out)
    return output_text

a=time.time()
output= generate()
b=time.time()
print(output)

print(f"total time it took to generate one response: {b-a}")