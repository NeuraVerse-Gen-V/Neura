# Configuration for the model training, and inference parameters
training = {
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "num_epochs": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_len": 512,
    "embedding_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "fp16": True,
    "scheduler": "cosine"
}

inference = {
    "max_length": 256,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.8,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1
}
