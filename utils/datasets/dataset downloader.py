from datasets import load_dataset

ds = load_dataset("SAGI-1/Greetings_DPO_dataset_V1")

# Save the 'train' split to CSV (change 'train' if needed)
ds["train"].to_csv("greetings.csv")