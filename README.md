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

### **Phase 4: Add sentiments**
* Add emotions to the Model
---
### **Phase 5: Vision Integration**

* Add image recognition capabilities
* Upgrade to live screen feed processing

---

# Project Structure

```
/PROJECT_EXHIBITION
├── main.py              # Main controller
├── model/
│   ├── transformer.py   # Encoder-decoder model
│   ├── tokenizer.py     # BPE tokenizer
│   └── inference.py     # generate() function
├── voice/
│   ├── stt.py           # Speech-to-text → tensor
│   ├── tts.py           # Text-to-speech
├── vision/
│   ├── screen_reader.py # Screen capture + OCR or CNN
│   └── preprocess.py    # Image → tensor
├── utils/
│   └── config.py        # JSON config loader/saver
```

---

# Data Flow Overview

```
[Image] ─► Image Encoder (CNN / ViT)
                  ↓
             Image Embeddings  ─┐
                                │
[Text Input] ─► BPE Tokenizer ─► Text Encoder ─┐
                                              │
[User Emotion ID] ─► Emotion Embedding ───────┤
[Bot Emotion ID ] ─► Emotion Embedding ───────┘
                     ↓
           Fusion Module / Attention
                     ↓
                  Decoder
                     ↓
             [Generated Response] ──┐
                                    ↓
                      Response Classifier (Toxic/Rude?)
                                    ↓
         ┌──────────────────────────┴────────────────────────────┐
         │ If safe                                                │
         │   → Return [Generated Response]                        │
         │ If rude                                                │
         │   → Reflex Generator (takes response as input)         │
         │        ↓                                               │
         │   → Generate [Apology/Correction]                      │
         │   → Return: [Generated Response + Correction]          │
         └────────────────────────────────────────────────────────┘

```
