import torch
from transformers import AutoTokenizer

# ==================== Model Configs ====================
max_len = 256        # Maximum generation length
d_model = 128        # Model embedding dimension
n_layers = 4         # Number of transformer layers
n_heads = 4          # Number of attention heads
ffn_hidden = 128     # Feedforward hidden layer size
drop_prob = 0.1      # Dropout probability

# =================== Training Configs ===================
batch_size = 128     # Training batch size
init_lr = 0.0005        # Initial learning rate
factor = 0.9         # Learning rate decay factor
patience = 10        # Early stopping patience
warmup = 100         # Warm-up steps
adam_eps = 5e-9      # Adam optimizer epsilon
epoch = 1000         # Number of training epochs
clip = 1             # Gradient clipping threshold
weight_decay = 5e-4  # L2 regularization (weight decay)

# =================== Tokenizer ===================
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add pad/eos tokens if they don’t exist in GPT-2 tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "bos_token": "<sos>"
    })

# =================== Dynamic Parameters =================
src_pad_idx = tokenizer.pad_token_id
trg_pad_idx = tokenizer.pad_token_id
trg_sos_idx = tokenizer.bos_token_id
eos_token   = tokenizer.eos_token_id

enc_voc_size = len(tokenizer)
dec_voc_size = len(tokenizer)
device = "cuda" if torch.cuda.is_available() else "cpu"
