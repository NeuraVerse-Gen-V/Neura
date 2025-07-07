import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
______Transformer Model Components______

Positional encoding
Multi head attention
scaled dot product attention
Layer norm
Position-wise feed forward network
Encoder layer
Encoder
Decoder layer
Decoder
Transformer embedding

"""


class PositionalEncoding(nn.Module):
    """
    Use sinusodial encoding
    """

    def __init__(self,d_model,max_len,device):
        """
        constructor for encoding
        d_model=dimensions
        max_len=max sequence lenght
        device = cuda or cpu
        """

        super(PositionalEncoding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad=False #to be tested with true

        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        #unsqueez to convert it from 1d to 2d to repesent words better

        idx = torch.arange(0,d_model,step=2,device=device).float()
        #idx is the index of d_model, eg embedding size=50 , idx=[0,50]
        #step=2 means idx is multiplied by 2

        self.encoding[:,0::2]=torch.sin(pos/10000**(idx/d_model))
        self.encoding[:,1::2]=torch.cos(pos/10000**(idx/d_model))

    def forward(self,x):
        #self encoding
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]
        # it will add with tok_emb : [batch_size, seq_len, d_model] 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        #calculate dot product with weight matrixes
        q,k,v=self.w_q(query), self.w_k(key), self.w_v(value)

        #split into tensors by n_heads
        q,k,v=self.split(q), self.split(k), self.split(v)

        #compute similarity
        out,attention= self.attention(q, k, v, mask=mask)

        out=self.concat()
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
            score=score.masked_fill(mask==0,float('-inf'))
        
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

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
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
    pass # Placeholder for the Transformer model implementation