import matplotlib.pyplot as plt
import json

with open("utils/log.json","r") as ri:
    data=json.load(ri)

epoches=[]
train_loss=[]
val_loss=[]
lr=[]
for epoch in data:
    epoches.append(epoch)
    train_loss.append(data[epoch]["train"])
    val_loss.append(data[epoch]["val"])
    lr.append(data[epoch]["lr"])

plt.title("Model Trainig Results")
plt.plot(epoches,train_loss,label="Train Loss")
plt.plot(epoches,val_loss,label="Val Loss")
plt.plot(epoches,lr,label="Learning Rate")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss and Learning Rate")
plt.show()