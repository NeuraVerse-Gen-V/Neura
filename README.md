Here’s a cleaned and more professional version of your `.md`:

---

# 🎓 Project Exhibition

This repository contains the AI chatbot project developed for the 2nd year exhibition.

---

# 🧠 AI Model Roadmap

### **Phase 1: Core Chatbot**

1. Build an encoder-decoder transformer (RAG-based with FAISS)
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

### **Phase 3: Vision Integration**

* Add image recognition capabilities
* Upgrade to live screen feed processing

---

# 🗂️ Project Structure

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

# 🔄 Data Flow Overview

```
[Image] ─► Image Encoder (CNN / ViT)
                  ↓
             Image Embeddings  ─┐
                                │
[Text Input] ─► BPE Tokenizer ─► Text Encoder ─┐
                                              │
                     ┌────────────────────────┘
                     │  Fusion Module (optional)
                     ↓
                  Decoder
                     ↓
             [Generated Response]
```
