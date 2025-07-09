from model.transformer import Transformer

from utils.config import *
from utils import dataloader

import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    eos_token_id=eos_token,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)



model.apply(initialize_weights)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)
scheduler.get_last_lr()
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

#load up data.csv for training
data=dataloader.load_data("utils/datasets/emotions_dataset.csv")
#convert loaded data into tensors
input_labels=data["input"]
output_labels=data["output"]

def train(input,output):
    pass #Training logic goes here

def evaluate(input):
    pass #Evaluation logic goes here

if __name__=="__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    inp_tensor,out_tensor=dataloader.tensorize(input_labels=input_labels,output_labels=output_labels)
    
    #train the model then eval it and then save the best model
    trained_model=train(inp_tensor,out_tensor)
    evaluate(trained_model)
    #Kirti do it, data will be in (input,output) format in a csv/excel file