# Project Exhibition
Here lies the project for 2nd year.




# AI MODEL ROADMAP
Things to make-

Phase 1-

1- Make the chatbot

a) make an encoder-decoder transformer model architecture (rag based with faiss)
b) make a custom dataset loader
c) BPE Tokenizer-(words to tokens, tokens to words, tokenizer trainer)
d) Make a trainer and an inference script
e) make a config.py to store the model configs

Phase 2-
a) Add speech recognition/input
b) Add speech output

Phase 3-
a) Give the model image recognition
b) Turn that image recognition into live feed



# STRUCTURE OF THE PROJECT

/PRODUCT EXHIBITION
├── main.py              # controller
├── model/
│   ├── transformer.py   # encoder-decoder model
│   ├── tokenizer.py     # BPE tokenizer
│   └── inference.py     # generate() function
├── voice/
│   ├── stt.py           # speech-to-text → tensor
│   ├── tts.py           # text-to-speech
├── vision/
│   ├── screen_reader.py # screen capture + OCR/CNN
│   └── preprocess.py    # image → tensor
├── utils/
│   └── config.py        # json config loader/saver
