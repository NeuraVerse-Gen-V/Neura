import torch
import aiohttp
import io
from PIL import Image

from model.transformer import Transformer
from model.tokenizer import BPETokenizer
from model.vision_encoder import transform  # <-- your preprocessing transform

from utils.config import *
import discord_interface.discord_interface as discord_conector

#-------------------------------------------------DISCORD SETUP-------------------------------------------------
discord_bot = discord_conector.BotClient()
bot = discord_bot.bot

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    img_tensor = None
    if message.attachments:
        attachment = message.attachments[0]
        if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        img_bytes = await resp.read()
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img_tensor = transform(image).unsqueeze(0).to(device)  # use your vision transform

    msg = await message.reply("Neura is thinking...")
    print(f"Received: {message.content}, Image: {img_tensor is not None}")


    output = generate(message.content, img_tensor)  # now supports both
    print(f"Generated: {output}")

    await msg.edit(content=output)
    await bot.process_commands(message)

#-------------------------------------------------MODEL SETUP-------------------------------------------------
tokenizer = BPETokenizer("gpt2")
model = Transformer(src_pad_idx, trg_pad_idx, trg_sos_idx, eos_token,
                    enc_voc_size, dec_voc_size, d_model, n_heads,
                    ffn_hidden, n_layers, drop_prob, device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

def generate(input_text, image=None):
    inp_tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(inp_tokens, max_len=20, images=image)  # <-- forward supports images
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
