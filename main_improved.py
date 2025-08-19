import torch
import torch.nn.functional as F
from model.transformer import Transformer
from model.tokenizer import BPETokenizer
from utils.config import *
import time

tokenizer = BPETokenizer("gpt2")
model = Transformer().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

def generate_with_sampling(input_text, max_len=50, temperature=1.0, top_k=50, top_p=0.9):
    """
    Generate text using temperature sampling, top-k, and nucleus (top-p) sampling
    """
    inp_tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    src_mask = model.make_src_mask(inp_tokens)
    
    # Start target sequence with <sos>
    trg_indices = [trg_sos_idx]
    
    for _ in range(max_len):
        trg_tensor = torch.tensor(trg_indices, dtype=torch.long, device=device).unsqueeze(0)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        # Forward pass
        enc_src = model.encoder(inp_tokens, src_mask)
        output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Get logits for last token
        logits = output[:, -1, :] / temperature  # Apply temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Stop if <eos>
        if next_token == eos_token:
            break
            
        trg_indices.append(next_token)
    
    # Remove <sos> and decode
    output_tokens = trg_indices[1:]
    try:
        output_text = tokenizer.decode(output_tokens)
        return output_text
    except:
        return f"Could not decode tokens: {output_tokens}"

def generate_beam_search(input_text, max_len=50, beam_size=3):
    """
    Generate text using beam search for better quality
    """
    inp_tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    src_mask = model.make_src_mask(inp_tokens)
    
    # Initialize beam with <sos>
    beams = [([trg_sos_idx], 0.0)]  # (sequence, score)
    
    for step in range(max_len):
        new_beams = []
        
        for sequence, score in beams:
            if sequence[-1] == eos_token:  # If sequence ended, keep as is
                new_beams.append((sequence, score))
                continue
                
            trg_tensor = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            # Forward pass
            enc_src = model.encoder(inp_tokens, src_mask)
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            # Get logits for last token
            logits = output[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get top beam_size tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_size)
            
            for i in range(beam_size):
                new_token = top_indices[0, i].item()
                new_score = score + top_log_probs[0, i].item()
                new_sequence = sequence + [new_token]
                new_beams.append((new_sequence, new_score))
        
        # Keep only top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Stop if all beams ended
        if all(seq[-1] == eos_token for seq, _ in beams):
            break
    
    # Return best sequence
    best_sequence, _ = beams[0]
    output_tokens = best_sequence[1:]  # Remove <sos>
    
    try:
        output_text = tokenizer.decode(output_tokens)
        return output_text
    except:
        return f"Could not decode tokens: {output_tokens}"

# Test different generation methods
if __name__ == "__main__":
    test_input = "who are you?"
    
    print("=== Testing Different Generation Methods ===")
    print(f"Input: '{test_input}'")
    print()
    
    # Method 1: Original greedy decoding
    print("1. Original Greedy Decoding:")
    a = time.time()
    try:
        inp_tokens = torch.tensor(tokenizer.encode(test_input), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(inp_tokens, max_len)
        output = tokenizer.decode(out)
        b = time.time()
        print(f"Output: '{output}'")
        print(f"Time: {b-a:.3f}s")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Method 2: Temperature sampling
    print("2. Temperature Sampling (temp=0.8):")
    a = time.time()
    output = generate_with_sampling(test_input, max_len, temperature=0.8, top_k=50, top_p=0.9)
    b = time.time()
    print(f"Output: '{output}'")
    print(f"Time: {b-a:.3f}s")
    print()
    
    # Method 3: Higher temperature
    print("3. Higher Temperature Sampling (temp=1.2):")
    a = time.time()
    output = generate_with_sampling(test_input, max_len, temperature=1.2, top_k=30, top_p=0.8)
    b = time.time()
    print(f"Output: '{output}'")
    print(f"Time: {b-a:.3f}s")
    print()
    
    # Method 4: Beam search
    print("4. Beam Search (beam_size=3):")
    a = time.time()
    output = generate_beam_search(test_input, max_len, beam_size=3)
    b = time.time()
    print(f"Output: '{output}'")
    print(f"Time: {b-a:.3f}s")
