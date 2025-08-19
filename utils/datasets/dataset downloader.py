import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/boltuix/emotions-dataset/emotions_dataset.parquet")

# Save as CSV
df.to_csv("emotions_dataset.csv", index=False)