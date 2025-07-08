import json
import torch

# ==================== Model Configs ====================
max_len = 256        # Maximum length of input sequence
d_model = 512        # Model embedding dimension
n_layers = 6         # Number of transformer layers
n_heads = 8          # Number of attention heads
ffn_hidden = 2048    # Feedforward hidden layer size
drop_prob = 0.1      # Dropout probability

# =================== Training Configs ===================
batch_size = 128     # Training batch size
init_lr = 0.1        # Initial learning rate
factor = 0.9         # Learning rate decay factor
patience = 10        # Early stopping patience
warmup = 100         # Warm-up steps
adam_eps = 5e-9      # Adam optimizer epsilon
epoch = 1000         # Number of training epochs
clip = 1             # Gradient clipping threshold
weight_decay = 5e-4  # L2 regularization (weight decay)


# =================== Dynamic Parameters =================
with open("model/vocab.json", "r") as f:
    vocab = json.load(f)
src_pad_idx = vocab["<pad>"]
trg_pad_idx = vocab["<pad>"]
trg_sos_idx = vocab["<sos>"]

enc_voc_size = len(vocab)
dec_voc_size = len(vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"