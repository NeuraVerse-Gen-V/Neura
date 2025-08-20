# analyzer.py
import torch
import time
import math
import tabulate
import psutil
from torch.utils.data import DataLoader, TensorDataset

from model.transformer import Transformer
from utils.config import *
from utils.dataloader import load_data, tensorize
from utils import graph   # <- your updated graph.py
import tabulate

save_dir = "utils/analysis/"

# ----------------------- LOAD MODEL -----------------------
checkpoint_path = "best_model.pt"
model = Transformer().to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("\n✅ Model loaded successfully!\n")

# ----------------------- PARAMETER ANALYSIS -----------------------
total_params, total_size = 0, 0
layer_info, layer_params = [], {}

for i, (name, param) in enumerate(model.named_parameters(), 1):
    count = param.numel()
    size_bytes = count * param.element_size()
    total_params += count
    total_size += size_bytes
    layer_info.append([
        i,
        name,
        f"{count:,}",
        f"{size_bytes/1024:.2f} KB",
        str(param.dtype).replace("torch.", "")
    ])
    layer_params[name] = count

print("📊 Model Parameter Breakdown\n")
print(tabulate.tabulate(
    layer_info,
    headers=["#", "Layer Name", "Param Count", "Size", "DType"],
    tablefmt="fancy_grid"
))

print("\n" + "="*60)
print(f"🔹 Total Parameters : {total_params:,}")
print(f"🔹 Estimated Size   : {total_size / (1024**2):.2f} MB")
print(f"🔹 Model Device     : {device}")
print(f"🔹 Loaded Checkpoint: {checkpoint_path}")
print("="*60 + "\n")

# ----------------------- MEMORY USAGE -----------------------
def memory_report():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")
    else:
        mem = psutil.Process().memory_info().rss / 1024**2
        print(f"CPU RAM Usage       : {mem:.2f} MB\n")

print("📦 Memory Report")
memory_report()

# ----------------------- BENCHMARK (GENERATION) -----------------------
def benchmark_generate(model, seq_len=32, max_len=50, n_runs=10):
    model.eval()
    inp_tokens = torch.randint(0, max_len, (1, seq_len)).to(device)

    # Warmup
    for _ in range(3):
        _ = model.generate(inp_tokens, max_len=max_len)

    # Timing
    start = time.time()
    for _ in range(n_runs):
        _ = model.generate(inp_tokens, max_len=max_len)
    end = time.time()

    avg_time = (end - start) / n_runs
    return avg_time, 1 / avg_time

print("⏱️ Benchmarking (text generation)...")
avg_time, throughput = benchmark_generate(model)
print(f"Avg Generation Time : {avg_time*1000:.2f} ms / sequence")
print(f"Throughput          : {throughput:.2f} sequences/sec\n")

# ----------------------- PERPLEXITY -----------------------
def evaluate_perplexity(model, dataloader, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            outputs = model(src, trg[:, :-1])   # [B, T-1, V]
            logits = outputs.reshape(-1, outputs.size(-1))
            targets = trg[:, 1:].reshape(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

val_loader = None
ppl = None
df_path = "utils/datasets/small_data.csv"
df = load_data(df_path)
if df is not None:
    inp, out = tensorize(df["input"], df["output"])
    dataset = TensorDataset(inp, out)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    ppl = evaluate_perplexity(model, val_loader, device)
    print(f"📉 Validation Perplexity: {ppl:.2f}\n")

# ----------------------- RUNTIME SCALING DATA -----------------------
seq_lengths = [8, 16, 32, 64, 128]
runtimes = []
for L in seq_lengths:
    dummy = torch.randint(0, 1000, (1, L)).to(device)
    start = time.time()
    _ = model.generate(dummy, max_len=50)
    runtimes.append((time.time() - start) * 1000)

# ----------------------- PLOT ALL GRAPHS -----------------------
graph.plot_runtime_scaling(seq_lengths, runtimes)
graph.plot_training()

# ----------------------- SAVE MARKDOWN REPORT -----------------------


with open(save_dir+"analysis_report.md", "w") as f:
    f.write("# Model Analysis Report\n\n")
    f.write(f"- Total Parameters: {total_params:,}\n")
    f.write(f"- Estimated Size  : {total_size / (1024**2):.2f} MB\n")
    f.write(f"- Device          : {device}\n")
    f.write(f"- Checkpoint      : {checkpoint_path}\n")
    f.write(f"- Avg Gen Time    : {avg_time*1000:.2f} ms\n")
    f.write(f"- Throughput      : {throughput:.2f} seq/s\n")
    if ppl is not None:
        f.write(f"- Validation PPL  : {ppl:.2f}\n")
    
    f.write("\n## Model Parameter Breakdown\n\n")
    # tabulate table in markdown format
    md_table = tabulate.tabulate(
        layer_info,
        headers=["#", "Layer Name", "Param Count", "Size", "DType"],
        tablefmt="github"
    )
    f.write(md_table + "\n\n")

    f.write("## Graphs\n\n")
    f.write(f"![Runtime](runtime_scaling.png)\n\n")
    f.write(f"![Training](training.png)\n\n")

print(f"✅ Analysis report saved to {save_dir}analysis_report.md")
