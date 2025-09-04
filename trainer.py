from model.transformer import Transformer

from utils.config import *
from utils import dataloader

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset,random_split

from tqdm import tqdm

import torch
import torch.nn as nn
import os
import gc


#load up data.csv for training
data=dataloader.load_data("utils/datasets/data.csv")
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
    
# Extract slices
input_labels  = data["input"][:no_of_lines]
output_labels = data["output"][:no_of_lines]
try:
    emotion_labels = data["emotion"][:no_of_lines]
except:
    emotion_labels=["Neutral" for i in range(len(input_labels))]
try:
    tone_labels   = data["tone"][:no_of_lines]
except:
    tone_labels= ["Neutral" for i in range(len(input_labels))]

output_labels = [
    f"{o} [EMOTION={e}] [TONE={t}]"
    for o, e, t in zip(output_labels, emotion_labels, tone_labels)
]

# Convert to tensors
inp_tensor, out_tensor = dataloader.tensorize(
    input_labels=input_labels,
    output_labels=output_labels
)

# Validation
if len(input_labels) != len(output_labels):
    raise ValueError(f"Input and output lengths don't match: {len(input_labels)} vs {len(output_labels)}")


model = Transformer().to(device)
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


def train_and_evaluate(model, input_tensor, output_tensor, clip, num_epochs=None, target_val_loss=1.0):
    # Default epochs if not specified
    if num_epochs is None:
        num_epochs = epochs

    # Split data into train/val (90/10 split)
    dataset = TensorDataset(input_tensor, output_tensor)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training"):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # Teacher forcing

            # Reshape for loss: (batch*seq, vocab)
            output = output.reshape(-1, output.shape[-1])
            tgt_y = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Scheduler step (with train loss)
            scheduler.step(loss.item())

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation"):
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[-1])
                tgt_y = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        if early_stopper.early_stop(avg_val_loss) or avg_val_loss <= target_val_loss:
            print("Early stopping triggered.")
            break

    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
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
    
    #cleanup before training
    data,emotion_labels,tone_labels
    # Train and evaluate the model with validation-based early stopping
    trained_model = train_and_evaluate(model=model, input_tensor=inp_tensor, output_tensor=out_tensor, clip=clip)
    
    # Save the final best model
    torch.save(trained_model.state_dict(), checkpoint_path)
    print(f"Best model saved as {checkpoint_path}")
