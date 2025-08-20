import matplotlib.pyplot as plt
import json

save_dir="utils/analysis/"
def plot_param_distribution(layer_params, save_path=save_dir+"param_distribution.png"):
    names = list(layer_params.keys())
    values = list(layer_params.values())

    plt.figure(figsize=(10,5))
    plt.bar(names, values, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Parameters")
    plt.title("Parameter Distribution per Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_runtime_scaling(lengths, times, save_path=save_dir+"runtime_scaling.png"):
    plt.figure(figsize=(6,4))
    plt.plot(lengths, times, marker="o")
    plt.xlabel("Sequence Length")
    plt.ylabel("Avg Time (ms)")
    plt.title("Runtime vs Sequence Length")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_perplexity(checkpoints, ppl_values, save_path=save_dir+"perplexity_over_time.png"):
    plt.figure(figsize=(6,4))
    plt.plot(checkpoints, ppl_values, marker="o", color="green")
    plt.xlabel("Checkpoint")
    plt.ylabel("Perplexity")
    plt.title("Perplexity over Checkpoints")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training():
    with open("utils/log.json", "r") as ri:
        data = json.load(ri)

    epoches = []
    train_loss = []
    val_loss = []
    lr = []

    for epoch in data:
        epoches.append(int(epoch))
        train_loss.append(data[epoch]["train"])
        val_loss.append(data[epoch]["val"])
        lr.append(data[epoch]["lr"])

    plt.figure(figsize=(10, 6))
    plt.title("Model Training Results")
    plt.plot(epoches, train_loss, label="Train Loss")
    plt.plot(epoches, val_loss, label="Val Loss")
    plt.plot(epoches, lr, label="Learning Rate")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Learning Rate")

    # Label only every 10th epoch
    plt.xticks([e for e in epoches if e % 10 == 0])

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir+"training.png")