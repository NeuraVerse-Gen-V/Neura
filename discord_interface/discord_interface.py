from discord.ext import commands

import discord


class BotClient:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='!', intents=intents)

        @self.bot.event
        async def on_ready():
            print(f'Logged in as {self.bot.user.name} - {self.bot.user.id}')

        @self.bot.command()
        async def join(ctx):
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                await channel.connect()
                await ctx.send(f'Joined {channel.name}')
            else:
                await ctx.send("You are not connected to a voice channel.")

        @self.bot.command()
        async def leave(ctx):
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                await ctx.send("Disconnected from the voice channel.")
            else:
                await ctx.send("I am not connected to any voice channel.")

        
    def run(self):
        token="MTQwNjI2MjAwNTgxOTUwNjY5OQ.GE4suL.ZcNoH4hY4_qUvoHBOVpMZnxy7kW0HHe3fwuYbY"
        self.bot.run(token)