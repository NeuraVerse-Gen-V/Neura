import torch
from model.transformer import Transformer
from model.tokenizer import BPETokenizer
from utils.config import *
#from vision import screen_reader,preprocess
#from voice import stt,tts

#-------------------------------------------------MODEL SETUP-------------------------------------------------
tokenizer=BPETokenizer()
model=Transformer().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

def generate(input):  
    inp_tokens = torch.tensor(tokenizer.encode(input), dtype=torch.long, device=device).unsqueeze(0)
    
    out=model.generate(inp_tokens,max_len=20)
    output_text = tokenizer.decode(out)
    return output_text

def voice_input():
    pass #code for voice input goes here

def vision():
    pass #code for vision goes here

def speak():
    pass #code for TTS goes here