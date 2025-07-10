from model.transformer import Transformer

from utils.config import *
from utils import dataloader

import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

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


def get_linear_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        return min((step + 1) / warmup_steps, 1)
    return LambdaLR(optimizer, lr_lambda)

def train(input_tensor, output_tensor):
    model.to(device)
    model.train()

    dataset = TensorDataset(input_tensor, output_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(params=model.parameters(),
                     lr=init_lr,
                     weight_decay=weight_decay,
                     eps=adam_eps)

    # Warmup scheduler first
    warmup_scheduler = get_linear_warmup_scheduler(optimizer, warmup)
    # After warmup, ReduceLROnPlateau for decay on plateau
    plateau_scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

    best_loss = float('inf')
    no_improve_epochs = 0

    global_step = 0
    for ep in tqdm(range(epoch), desc="Epoch:"):
        epoch_loss = 0.0
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Step warmup scheduler while warming up
            if global_step < warmup:
                warmup_scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        plateau_scheduler.step(avg_loss)

        print(f"Epoch {ep+1}/{epoch}, Loss: {avg_loss:.4f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs > patience:
                print("Early stopping triggered.")
                break

    return model



def evaluate(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        sample = torch.randint(1, enc_voc_size, (1, max_len)).to(device)
        output = model(sample)
        prediction = output.argmax(dim=-1)
        print("Sample output token IDs:", prediction[0].tolist())


if __name__=="__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    inp_tensor,out_tensor=dataloader.tensorize(input_labels=input_labels,output_labels=output_labels)
    
    #train the model then eval it and then save the best model
    trained_model=train(inp_tensor,out_tensor)
    evaluate(trained_model)