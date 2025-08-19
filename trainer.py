from model.transformer import Transformer

from utils.config import *
from utils import dataloader

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from model.vision_encoder import transform

import torch
import torch.nn as nn
import json
import os
"""
TODO:
- Add images handeling into the training loop
-- to do that you first need to update the dataloader to handle images
--- if images col is missing then skip image processing and set images to None
---- if some images are present then set the empty values to None
"""

#load up data.csv for training
data=dataloader.load_data("utils/datasets/img.csv")
if data is None:
    raise ValueError("Failed to load training data")

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
    
#convert loaded data into tensors
input_labels=data["input"][:no_of_lines]
output_labels=data["output"][:no_of_lines]
image = data["image"][:no_of_lines] if "image" in data.columns else None

inp_tensor,out_tensor,image_tensor=dataloader.tensorize(input_labels=input_labels,output_labels=output_labels,image_paths=image,transform=transform)

# Validate data
if len(input_labels) != len(output_labels):
    raise ValueError(f"Input and output lengths don't match: {len(input_labels)} vs {len(output_labels)}")


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    eos_token_id=eos_token,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

#freeze the vision encoder parameters
print("Freezing vision encoder parameters...")
for name, param in model.named_parameters():
    if name.startswith("vision_encoder"):
        param.requires_grad = False
print("Vision encoder parameters frozen.")
model.apply(initialize_weights)

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


def train_and_evaluate(model, input_tensor, output_tensor, image_tensor, clip, num_epochs=None, target_val_loss=1.0):
    if num_epochs is None:
        num_epochs = epoch  # from config
    
    # --- Dataset setup ---
    if image_tensor is not None:
        processed_images = []
        for img in image_tensor:
            if img is None:
                processed_images.append(torch.zeros(3, size_of_image, size_of_image))  # dummy
            else:
                processed_images.append(img)
        dataset = TensorDataset(input_tensor, output_tensor, torch.stack(processed_images))
        use_images = True
    else:
        dataset = TensorDataset(input_tensor, output_tensor)
        use_images = False

    # --- Dataloaders ---
    effective_batch_size = min(batch_size, len(dataset))
    if effective_batch_size < batch_size:
        print(f"⚠️ Dataset size ({len(dataset)}) < batch size ({batch_size}), using {effective_batch_size}")
    
    train_dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    val_dataloader   = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, drop_last=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Training until val_loss ≤ {target_val_loss}")
    
    with open("utils/log.json","r") as ri:
        logs = json.load(ri)
    
    for epoch_idx in tqdm(range(num_epochs), desc="Training model"):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader)):
            if use_images:
                src, trg, images = batch
                images = images.to(device)
            else:
                src, trg = batch
                images = None
            
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1], images)
            
            loss = criterion(output.contiguous().view(-1, output.shape[-1]),
                             trg[:, 1:].contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", total=len(val_dataloader)):
                if use_images:
                    src, trg, images = batch
                    images = images.to(device)
                else:
                    src, trg = batch
                    images = None

                src, trg = src.to(device), trg.to(device)
                output = model(src, trg[:, :-1], images)
                loss = criterion(output.contiguous().view(-1, output.shape[-1]),
                                 trg[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss   = val_loss / len(val_dataloader)
        current_lr     = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch_idx+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        logs[str(epoch_idx+1)] = {"train": avg_train_loss, "val": avg_val_loss, "lr": current_lr}
        with open("utils/log.json", "w") as wi:
            json.dump(logs, wi, indent=4)

        # --- Save best ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"✅ New best model (val_loss={best_val_loss:.4f})")

        scheduler.step(avg_val_loss)

        if avg_val_loss <= target_val_loss:
            print(f"🎯 Target val_loss {target_val_loss} reached at epoch {epoch_idx+1}")
            break
        if early_stopper.early_stop(avg_val_loss):
            print(f"⏹️ Early stopping at epoch {epoch_idx+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"🔄 Loaded best model (val_loss={best_val_loss:.4f})")
    
    return model


if __name__=="__main__":
    #load the model weights if available
    checkpoint_path = "best_model.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint, continuing training...")
    else:
        print("No checkpoint found, starting training from scratch.")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    #print(inp_tensor,out_tensor)
    
    # Train and evaluate the model with validation-based early stopping
    trained_model = train_and_evaluate(model=model, input_tensor=inp_tensor, output_tensor=out_tensor,image_tensor=image_tensor, clip=clip)
    
    # Save the final best model
    torch.save(trained_model.state_dict(), checkpoint_path)
    print(f"Best model saved as {checkpoint_path}")
