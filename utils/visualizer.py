from graphviz import Digraph

import os
from graphviz import Digraph

# Point Python to Graphviz bin
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Create layered directed graph (top -> bottom) with color-coded layers
dot = Digraph(comment="Color-Coded Layered Project Architecture", format="png")
dot.attr(rankdir="TB", size="10")

# Layer 1: Base utilities (red-ish)
dot.node("config.py", "config.py\n(Hyperparameters, tokenizer, device info)", shape="box", style="filled", fillcolor="#ffdddd")
dot.node("tokenizer.py", "tokenizer.py\n(Text encoding/decoding)", shape="box", style="filled", fillcolor="#ffdddd")
dot.node("vision_encoder.py", "vision_encoder.py\n(Image → Embeddings)", shape="box", style="filled", fillcolor="#ffdddd")

# Layer 2: Core components (yellow-ish)
dot.node("dataloader.py", "dataloader.py\n(Load dataset, tokenize text,\ntransform images → tensors)", shape="box", style="filled", fillcolor="#fff0b3")
dot.node("transformer.py", "transformer.py\n(Core model: Encoder, Decoder,\nVisionEncoder, masks, inference)", shape="box", style="filled", fillcolor="#fff0b3")

# Layer 3: Training & Interfaces (green-ish)
dot.node("trainer.py", "trainer.py\n(Training loop, validation,\ncheckpointing, LR scheduling)", shape="box", style="filled", fillcolor="#d5f5e3")
dot.node("discord_interface.py", "discord_interface.py\n(Discord bot events & messaging)", shape="box", style="filled", fillcolor="#d5f5e3")

# Layer 4: Entry point & visualization (blue-ish)
dot.node("main.py", "main.py\n(Entry point: loads model,\nprocesses messages/images,\nruns Discord bot)", shape="box", style="filled", fillcolor="#cce5ff")
dot.node("graph.py", "graph.py\n(Visualizes loss/LR from utils/log.json)", shape="box", style="filled", fillcolor="#cce5ff")

# Subgraph rank for alignment in layers
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("config.py")
    s.node("tokenizer.py")
    s.node("vision_encoder.py")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("dataloader.py")
    s.node("transformer.py")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("trainer.py")
    s.node("discord_interface.py")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("main.py")
    s.node("graph.py")

# Define edges based on imports/usage
dot.edges([("config.py", "transformer.py"), ("config.py", "dataloader.py"), ("config.py", "trainer.py")])
dot.edges([("tokenizer.py", "dataloader.py"), ("tokenizer.py", "transformer.py")])
dot.edges([("vision_encoder.py", "transformer.py"), ("vision_encoder.py", "trainer.py")])
dot.edge("dataloader.py", "trainer.py")
dot.edges([("transformer.py", "trainer.py"), ("transformer.py", "main.py")])
dot.edge("discord_interface.py", "main.py")
dot.edge("graph.py", "main.py")

# Export diagram
dot.render("project_architecture_layered_colored", view=True)
