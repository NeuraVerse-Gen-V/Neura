import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.vision_encoder import VisionEncoder
"""
______Transformer Model Components______

Positional encoding
Multi head attention
scaled dot product attention
Layer norm
Position-wise feed forward network
Encoder layer
Decoder layer
TransformerEmbedding
Encoder
Decoder
Transformer

"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.d_model = d_model
        self.device = device

    def forward(self, x):
        # x: [batch_size, seq_len, d_model] or [batch_size, seq_len]
        batch_size, seq_len = x.size(0), x.size(1)
        pos = torch.arange(0, seq_len, device=self.device).float().unsqueeze(1)  # [seq_len,1]
        idx = torch.arange(0, self.d_model, 2, device=self.device).float()       # [d_model/2]

        pe = torch.zeros(seq_len, self.d_model, device=self.device)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (idx / self.d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (idx / self.d_model)))

        # expand to batch: [batch_size, seq_len, d_model]
        return pe.unsqueeze(0).expand(batch_size, -1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_head
        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        #calculate dot product with weight matrixes
        q,k,v=self.w_q(q), self.w_k(k), self.w_v(v)

        #split into tensors by n_heads
        q,k,v=self.split(q), self.split(k), self.split(v)

        #compute similarity
        out,attention= self.attention(q, k, v, mask=mask)

        out=self.concat(out)
        out=self.w_concat(out)

        return out
    
    def split(self, tensor):
        """
        split the tensor by n_heads

        - param tensor: [batch_size, seq_len, d_model]
        - return: [batch_size,head,lenght,d_tensor]
        """
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_heads
        tensor= tensor.view(batch_size, seq_len, self.n_heads, d_tensor).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_tensor]
        return tensor
    
    def concat(self, tensor):
        """
        inverse of split
        - param tensor: [batch_size, n_heads, seq_len, d_tensor]
        - return: [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_tensor = tensor.size()
        d_model = n_heads * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return tensor

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot product attention
    Query: given sentence (decoder)
    key: every sentence to check relationship with query (encoder)
    value: every sentence with the same key
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask=None,e=1e-12):
        batch_size,head,lenght,d_tensor=k.size()
        #calculate similarity
        k_transpose=k.transpose(2,3)
        score=(q@k_transpose)/math.sqrt(d_tensor)

        #used to hide the future tokens in decoder
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        #set values to 0 or 1
        score=self.softmax(score)

        #use residual connection of value
        v= score@v

        return v, score

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean=x.mean(-1,keepdim=True)
        var = x.var(-1,unbiased=False,keepdim=True)
        # -1 if for the last dimension. Eg- [1,2,3,4] -> 4 is the last dimension
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta

        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        residual_x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        
        # 3. positionwise feed forward network
        residual_x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):    
        # 1. compute self attention
        residual_x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            residual_x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + residual_x)

        # 5. positionwise feed forward network
        residual_x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + residual_x)
        return x

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)


class TransformerEmbedding(nn.Module):
    """
    Embedding layer for transformer
    """

    def __init__(self, d_model, vocab_size,padding_idx, drop_prob=0.1, device='cpu'):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEncoding(d_model=d_model, device=device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # x is the input token ids
        tok_emb = self.tok_emb(x)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x)   # [seq_len, d_model]
        return self.dropout(tok_emb + pos_emb)

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, d_model, src_pad_idx, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        padding_idx=src_pad_idx,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, src, mask=None, vision_embeds=None):
        # embed text
        x = self.emb(src)  # (batch, seq_len, d_model)
        # concat vision embeddings if provided
        if vision_embeds is not None:
            x = torch.cat([x, vision_embeds], dim=1)  # (batch, seq_len+vis_len, d_model)
        # pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        return x



class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, ffn_hidden,src_pad_idx, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        padding_idx=src_pad_idx,
                                        drop_prob=drop_prob,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, eos_token_id,
                 enc_voc_size, dec_voc_size, d_model, n_head,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.eos_token   = eos_token_id
        self.device = device

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            backbone="vit",
            model_name="google/vit-base-patch16-224-in21k",
            d_model=d_model,
            device=device
        )

        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            src_pad_idx=src_pad_idx,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            src_pad_idx=src_pad_idx,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

    def make_src_mask(self, src, vision_embeds=None):
        """
        src: (B, src_len)
        vision_embeds: (B, vis_len, d_model) or None
        """
        batch_size, src_len = src.size()
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,src_len)

        if vision_embeds is not None:
            vis_len = vision_embeds.size(1)
            vision_mask = torch.ones(batch_size, 1, 1, vis_len, device=src.device)
            mask = torch.cat([mask, vision_mask], dim=-1)  # (B,1,1,src_len+vis_len)

        return mask

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.size()
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,trg_len)

        causal_mask = torch.tril(torch.ones(trg_len, trg_len, device=self.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,trg_len,trg_len)

        return trg_pad_mask & causal_mask

    def forward(self, src, trg, images=None):
        vision_embeds = None
        if self.vision_encoder is not None and images is not None:
            vision_embeds = self.vision_encoder(images)

        src_mask = self.make_src_mask(src, vision_embeds)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask, vision_embeds=vision_embeds)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    @torch.no_grad()
    def generate(self, inp_tokens, max_len=50, images=None):
        # Get vision embeddings if images are provided
        vision_embeds = None
        if self.vision_encoder is not None and images is not None:
            vision_embeds = self.vision_encoder(images)

        # Build source mask with vision tokens included
        src_mask = self.make_src_mask(inp_tokens, vision_embeds)

        # Start target sequence with <sos>
        trg_indices = [self.trg_sos_idx]
        for _ in range(max_len):
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long, device=self.device).unsqueeze(0)
            trg_mask = self.make_trg_mask(trg_tensor)

            # Encode text + optional vision
            enc_src = self.encoder(inp_tokens, src_mask, vision_embeds=vision_embeds)
            output = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # Pick next token
            next_token = output[:, -1, :].argmax(-1).item()
            if next_token == self.eos_token:
                break
            trg_indices.append(next_token)

        return trg_indices[1:]  # remove <sos>

