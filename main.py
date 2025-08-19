import torch
from model.transformer import Transformer
from model.tokenizer import BPETokenizer

from utils.config import *
import discord_interface.discord_interface as discord_conector

#-------------------------------------------------DISCORD SETUP-------------------------------------------------
discord_bot = discord_conector.BotClient()
bot = discord_bot.bot

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Remove all image handling
    msg = await message.reply("Neura is thinking...")
    print(f"Received: {message.content}")

    output = generate(message.content)  # Only text
    print(f"Generated: {output}")

    await msg.edit(content=output)
    await bot.process_commands(message)

#-------------------------------------------------MODEL SETUP-------------------------------------------------
tokenizer = BPETokenizer("gpt2")
model = Transformer().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

def generate(input_text):
    inp_tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(inp_tokens, max_len=20)  # No images
    return tokenizer.decode(out)

def voice_input():
    pass

def vision():
    pass

def speak():
    pass

def discord_int():
    pass

if __name__ == "__main__":
    discord_bot.run()
