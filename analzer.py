from model.transformer import Transformer
from utils.config import *

"""
This file calculates the total number of parameters, their estimated file size,
and whether they are trainable (unlocked) or frozen (locked).
"""
model = Transformer().to(device)

total_params = 0
total_size = 0

for i, (name, param) in enumerate(model.named_parameters(), 1):
    count = param.numel()
    size_bytes = count * param.element_size()
    total_params += count
    total_size += size_bytes
    status = "unlocked" if param.requires_grad else "locked"
    print(f"param{i} {name}: {count} params, {size_bytes / 1024:.2f} KB, {status}")

print("-" * 50)
print(f"Total parameters: {total_params}")
print(f"Estimated size  : {total_size / (1024 ** 2):.2f} MB")
