import torch

from model.transformer import Transformer
from model.tokenizer import BPETokenizer
from utils.config import *
#from vision import screen_reader,preprocess
#from voice import stt,tts
import discord_interface.discord_interface as discord_conector

#-------------------------------------------------DISCORD SETUP-------------------------------------------------
discord_bot = discord_conector.BotClient()
bot = discord_bot.bot

@bot.event
async def on_message(message):
    if message.author.bot:  # avoid replying to itself or other bots
        return

    print(f"Received: {message.content}")
    output = generate(message.content)
    print(f"Generated: {output}")

    await message.reply(output)

    # Ensure commands still work
    await bot.process_commands(message)
#-------------------------------------------------MODEL SETUP-------------------------------------------------
tokenizer=BPETokenizer("gpt2")
model=Transformer(src_pad_idx,trg_pad_idx,trg_sos_idx,eos_token,enc_voc_size,dec_voc_size,d_model,n_heads,ffn_hidden,n_layers,drop_prob,device)
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

def discord_int():
    pass



if __name__ == "__main__":
    discord_bot.run()