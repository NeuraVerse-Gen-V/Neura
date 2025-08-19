import pandas as pd
import random
from datasets import load_dataset
import os
from tqdm import tqdm

# Load dataset
ds = load_dataset("peteole/coco2017-segmentation-10k-256x256")
num_samples = len(ds["train"])
save_dir = "utils/images"
os.makedirs(save_dir, exist_ok=True)

ai_queries = [
    # Objects / things
    "What is this?",
    "Can you identify this?",
    "What am I looking at?",
    "Describe this for me.",
    "What does this image contain?",
    "Can you tell me what this is?",
    "What object is this?",
    "Identify this item.",
    "What kind of object is this?",
    "Give me details about this.",
    "Explain what this is.",
    "What is shown here?",
    "Recognize this object.",
    "What could this be?",
    "Tell me what this depicts.",

    # Scenes / context
    "What do you see in this?",
    "Describe what’s happening here.",
    "What’s in this picture?",
    "Can you explain this scene?",
    "What’s going on here?",
    "Give me a description of this image.",
    "Explain the situation in this image.",
    "What is happening in this scene?",
    "Summarize what you see here.",
    "What is the context of this image?",
    "Can you describe the environment?",
    "What’s occurring in this photo?",
    "Give an overview of this scene.",

    # Conversational / casual
    "Hey, what’s this?",
    "Any idea what I’m looking at?",
    "Who’s in this?",
    "What’s happening here?",
    "Can you tell me what I’m seeing?",
    "Do you know what this is?",
    "What am I seeing right now?",
    "Give me info on this.",
    "What do you make of this?",
    "Do you recognize this?",
    "What’s this all about?",
    "Can you explain this to me?",
    "Any clues about what this is?",
    "Who or what is this?"
]



rows = []
for i in tqdm(range(num_samples),total=num_samples, desc="Processing dataset"):
    data = ds["train"][i]
    
    # Save the image once
    img_path = os.path.join(save_dir, f"img_{i}.png")
    data["image"].save(img_path)
    
    # Create rows for each caption
    for caption in data["captions"]:
        query = random.choice(ai_queries)
        rows.append({
            "input": query,
            "image": img_path,  # all captions point to the same image
            "output": caption
        })

# Save CSV
df = pd.DataFrame(rows)
df.to_csv("utils/datasets/img.csv", index=False)
print("Saved CSV and images to disk.")