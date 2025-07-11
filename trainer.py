from model.transformer import Transformer

from utils.config import *
from utils import dataloader

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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



#model.apply(initialize_weights)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)
#scheduler.get_last_lr()
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

#load up data.csv for training
data=dataloader.load_data("utils/datasets/data.csv")
if data is None:
    raise ValueError("Failed to load training data")
    
#convert loaded data into tensors
input_labels=data["input"][:1000]
output_labels=data["output"][:1000]

# Validate data
if len(input_labels) != len(output_labels):
    raise ValueError(f"Input and output lengths don't match: {len(input_labels)} vs {len(output_labels)}")
    
print(f"Loaded {len(input_labels)} training samples")

def train(model, input_tensor, output_tensor, clip, num_epochs=None):
    if num_epochs is None:
        num_epochs = epoch  # Use global epoch from config
    
    model.train()
    
    # Create dataset and dataloader for proper batching
    dataset = TensorDataset(input_tensor, output_tensor)
    
    # Adjust batch size if dataset is too small
    effective_batch_size = min(batch_size, len(dataset))
    if effective_batch_size < batch_size:
        print(f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({batch_size}). Using batch size {effective_batch_size}")
    
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    
    for epoch_idx in tqdm(range(num_epochs),desc="Training model"):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            
            """
            Add training code, only src,trg are needed, batch_idx is not needed.
            Let batch_idx be there or code WILL BREAK
            """
        
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f'Epoch {epoch_idx+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}')
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)

    #return the model after training
    return model

def evaluate(model):
    pass #Evaluation logic goes here

if __name__=="__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    inp_tensor,out_tensor=dataloader.tensorize(input_labels=input_labels,output_labels=output_labels)
    
    #train the model then eval it and then save the best model
    trained_model=train(model=model, input_tensor=inp_tensor, output_tensor=out_tensor, clip=clip)
    evaluate(trained_model)

    torch.save(model.state_dict(),"model.pt")