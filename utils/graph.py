import matplotlib.pyplot as plt
import json

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
plt.show()
