batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
init_lr = 0.1
factor = 0.9
patience = 10
warmup = 100
adam_eps = 5e-9
epoch = 1000
clip = 1
weight_decay = 5e-4

#dynamic parameters
import json
import torch
with open("../model/vocab.json", "r") as f:
    vocab = json.load(f)
src_pad_idx = vocab["<pad>"]
trg_pad_idx = vocab["<pad>"]
trg_sos_idx = vocab["<sos>"]

enc_voc_size = len(vocab)
dec_voc_size = len(vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"