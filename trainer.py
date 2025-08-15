from model.transformer import Transformer

from utils.config import *
from utils import dataloader

import torch
import torch.nn as nn
import json

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

class WarmupScheduler:
    """Custom learning rate scheduler with warmup functionality"""
    def __init__(self, optimizer, d_model, warmup_steps, factor=0.9, lr_patience=5):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.lr_patience = lr_patience
        self.step_num = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.lr_patience_counter = 0
        
    def step(self, loss=None):
        self.step_num += 1
        
        # Calculate learning rate with warmup
        if self.step_num <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # Use plateau reduction after warmup for learning rate scheduling
            if loss is not None:
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.lr_patience_counter = 0
                else:
                    self.lr_patience_counter += 1
                    if self.lr_patience_counter >= self.lr_patience:
                        self.base_lr *= self.factor
                        self.lr_patience_counter = 0
                        print(f"Learning rate reduced to {self.base_lr:.6f}")
            lr = self.base_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class EarlyStopper:
    """Early stopping utility using patience parameter"""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def early_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        return False

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

# Use custom warmup scheduler that incorporates warmup parameter from config
scheduler = WarmupScheduler(optimizer=optimizer,
                           d_model=d_model,
                           warmup_steps=warmup,
                           factor=factor,
                           lr_patience=5)  # Separate patience for LR scheduling

# Early stopping uses the patience parameter from config
early_stopper = EarlyStopper(patience=patience, min_delta=0.001)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

#load up data.csv for training
data=dataloader.load_data("utils/datasets/small_data.csv")
if data is None:
    raise ValueError("Failed to load training data")
    
#convert loaded data into tensors
input_labels=data["input"][:1000]
output_labels=data["output"][:1000]

# Validate data
if len(input_labels) != len(output_labels):
    raise ValueError(f"Input and output lengths don't match: {len(input_labels)} vs {len(output_labels)}")

def train_and_evaluate(model, input_tensor, output_tensor, clip, num_epochs=None, target_val_loss=1.0):
    if num_epochs is None:
        num_epochs = epoch  # Use global epoch from config
    
    # Create dataset and dataloader for proper batching
    dataset = TensorDataset(input_tensor, output_tensor)
    
    # Adjust batch size if dataset is too small
    effective_batch_size = min(batch_size, len(dataset))
    if effective_batch_size < batch_size:
        print(f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({batch_size}). Using batch size {effective_batch_size}")
    
    train_dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, drop_last=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Training will stop when validation loss reaches {target_val_loss} or below")
    
    with open("utils/log.json","r") as ri:
        logs=json.load(ri)
    for epoch_idx in tqdm(range(num_epochs), desc="Training model"):
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        for batch_idx, (src, trg) in tqdm(enumerate(train_dataloader),desc="Training ",total=len(train_dataloader)):
            src = src.to(device)
            trg = trg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output_reshape, trg_reshape)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (src, trg) in tqdm(enumerate(val_dataloader),desc="Evaluating ",total=len(val_dataloader)):
                src = src.to(device)
                trg = trg.to(device)
                
                # Forward pass
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg_reshape = trg[:, 1:].contiguous().view(-1)
                
                # Calculate loss
                loss = criterion(output_reshape, trg_reshape)
                val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch_idx+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.6f}')
        
        logs[str(epoch_idx+1)]={"train":avg_train_loss,"val":avg_val_loss,"lr":current_lr}
        with open("utils/log.json","w") as wi:
            json.dump(logs,wi,indent=4)
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Check if target validation loss is reached
        if avg_val_loss <= target_val_loss:
            print(f"Target validation loss {target_val_loss} reached! Stopping training at epoch {epoch_idx+1} with val loss: {avg_val_loss:.4f}")
            break
        
        # Check for early stopping using validation loss and patience parameter from config
        if early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping triggered after {epoch_idx+1} epochs. No improvement in validation loss for {patience} epochs.")
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model

if __name__=="__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    inp_tensor,out_tensor=dataloader.tensorize(input_labels=input_labels,output_labels=output_labels)
    
    # Train and evaluate the model with validation-based early stopping
    trained_model = train_and_evaluate(model=model, input_tensor=inp_tensor, output_tensor=out_tensor, clip=clip)
    
    # Save the final best model
    torch.save(trained_model.state_dict(), "best_model.pt")
    print("Best model saved as 'best_model.pt'")
