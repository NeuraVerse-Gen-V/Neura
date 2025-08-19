---

# Neura

This repository contains the AI Vtuber named Neura

---

# AI Model Roadmap

### **Phase 1: Core Chatbot**

1. Build an encoder-decoder transformer
2. Develop a custom dataset loader
3. Implement a BPE Tokenizer

   * Tokenization: words ⇄ tokens
   * Trainer for tokenizer
4. Create a training and inference script
5. Add `config.py` for model configurations

---

### **Phase 2: Voice Interface**

* Add speech-to-text (STT) input
* Add text-to-speech (TTS) output

---
### **Phase 3: (RAG-based with FAISS)**
* Somehow add Rag capablities to the model
* Add FAISS for faster RAG

---
### **Phase 4: Add sentiments**
Add Emotion Embeddings

   * For user mood and bot mood
   * Inject into decoder or fusion

Condition response on emotion
   * Bot replies happily/sadly etc.

Build Reflex Generator Module
   * Custom train the model to generate appopriate responses

Update Inference Pipeline
   * If bot says something flagged → run reflex generator → append apology
---
### **Phase 5: Vision Integration**

* Add image recognition capabilities
* Upgrade to live screen feed processing

---

# Project Structure

```
/PROJECT_EXHIBITION
├── main.py                     # Orchestrator: runs vision + model + voice
├── trainer.py                  # Training loop for text + multimodal
│
├── model/
│   ├── transformer.py          # Encoder–decoder backbone
│   ├── vision_encoder.py       # Wrapper: ViT/Swin/ConvNeXt → embeddings
│   ├── tokenizer.py            # Custom BPE
│   ├── inference.py            # generate(), beam search, structured response
│   └── reflex_generator.py     # Optional rule-based fallback
│
├── voice/
│   ├── stt.py                  # Speech → text
│   └── tts.py                  # Text → speech
│
├── vision/
│   ├── screen_reader.py        # Webcam / screen capture
│   ├── preprocess.py           # Resize, normalize, augmentation
│   └── ocr.py                  # (optional) text from screen for context
│
├── utils/
│   ├── config.py               # JSON config handler
│   ├── emotion_control.py      # Inject tokens to control bot mood
│   ├── output_parser.py        # Validate chatbot output (JSON/text)
│   └── datasets/               # Dataset loaders (text / multimodal)


```

---

# Data Flow Overview

![Image](./utils/Dataflow.png)

---

```
[Image] ─► Image Encoder (CNN / ViT)
                  ↓
             Image Embeddings  ─┐
                                │
[Text Input] ─► BPE Tokenizer ─►Text Encoder ─┐
                                              │
[User Emotion ID] ─► Emotion Embedding ───────┤
[Bot Emotion ID ] ─► Emotion Embedding ───────┘
                                              ↓
                                  Fusion Module / Attention
                                              ↓
                                           Decoder
                                              ↓
     ┌─────────────────────────────────────────────────────────┐
     │     Structured Output (as JSON):                        │
     │                                                         │
     │  {                                                      │
     │    "output": "I'm sorry, that was rude of me.",         │
     │    "emotion": "ashamed",                                │
     │    "motion": { "expression": "sad", "intensity": 0.8 }, │
     │    "toggles": { "sweat": true }                         │
     │  }                                                      │
     └─────────────────────────────────────────────────────────┘
                     ↓
      Send output to VTube Studio / UI / Speech TTS

```

---
# Model Output Format

Normal response:
```
Input:
"What is your name?"

Output:
{
  "output": "My name is josh!",
  "emotion": "Excited",
  "motion": { "expression": "excited", "intensity": 0.8 },
  "toggles": { "sweat": False,"Excited":True }
}

```

Reflex response:

```
Input:
"You made a mistake!"

Output:
{
  "output": "I'm sorry, that was rude of me.",
  "emotion": "ashamed",
  "motion": { "expression": "sad", "intensity": 0.8 },
  "toggles": { "sweat": true }
}
```

---
# Ideal HyperParams

```python
# General
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
batch_size = 64     # Training batch size
init_lr = 0.0005        # Initial learning rate
factor = 0.9         # Learning rate decay factor
patience = 10        # Early stopping patience
warmup = 100         # Warm-up steps
adam_eps = 5e-9      # Adam optimizer epsilon
epoch = 1000         # Number of training epochs
clip = 1             # Gradient clipping threshold
weight_decay = 5e-4  # L2 regularization (weight decay)
no_of_lines=10000    # Number of lines to read from the dataset
size_of_image=224    # Size of the input image for vision tasks

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

```
