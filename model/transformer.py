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
    
    def forward(self,q,k,v,mask=None):
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
    pass # Placeholder for position-wise feed forward network implementation

class EncoderLayer(nn.Module):
    pass # Placeholder for encoder layer implementation

class Encoder(nn.Module):
    pass # Placeholder for encoder implementation

class DecoderLayer(nn.Module):
    pass # Placeholder for decoder layer implementation

class Decoder(nn.Module):
    pass # Placeholder for decoder implementation

class Transformer(nn.Module):
    pass # Placeholder for the Transformer model implementation