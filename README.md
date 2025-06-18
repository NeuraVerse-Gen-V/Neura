---

# Project Exhibition

This repository contains the AI chatbot project developed for the 2nd year exhibition.

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
├── main.py                     # Main controller
├── model/
│   ├── transformer.py          # Encoder-decoder model
|   ├── vision_encoder.py       # ViT or CNN encoder for image → embedding
│   ├── tokenizer.py            # BPE tokenizer
│   ├── inference.py            # generate() + structured output
│   └── reflex_generator.py     # (Optional fallback generator)
├── voice/
│   ├── stt.py                  # Speech-to-text → tensor
│   └── tts.py                  # Text-to-speech
├── vision/
│   ├── screen_reader.py        # Screen capture + OCR or CNN
│   └── preprocess.py           # Image → tensor
├── utils/
│   ├── config.py               # JSON config loader/saver
│   ├── emotion_control.py      # Inject user/bot emotion tokens
│   ├── output_parser.py        # Parse & validate model JSON output
│   └── datasets/               # Dataset loaders, preprocessors, and splitters


```

---

# Data Flow Overview

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

---

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