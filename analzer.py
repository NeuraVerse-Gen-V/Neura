from model.transformer import Transformer
from utils.config import *

"""
This file is just to calculate the total number of Parameters this model has
and calculate the estimated file size of the model
"""
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

total_params = 0
total_size = 0

for i, (name, param) in enumerate(model.named_parameters(), 1):
    count = param.numel()
    size_bytes = count * param.element_size()
    total_params += count
    total_size += size_bytes
    print(f"param{i} {name}: {count} params, {size_bytes / 1024:.2f} KB")

print("-" * 50)
print(f"Total parameters: {total_params}")
print(f"Estimated size  : {total_size / (1024 ** 2):.2f} MB")

