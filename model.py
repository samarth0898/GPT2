import torch.nn.functional as F
import torch.nn as nn 
import torch
import math

class CausalSelfAttention(nn.Module):
    """
    Slightly more efficient implementation of the mulit-head attention block 

    Typical Mulit-head Attention Block 
    qkv from linear layer 
    
    """ 
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd,  dim = 2)
        k = k.view(B, T , self.n_head, C // self.n_embd).transpose(1, 2) # (B, nh, T, C)
        q = q.view(B, T , self.n_head, C // self.n_embd).transpose(1, 2) # (B, nh, T, C)
        v = v.view(B, T , self.n_head, C // self.n_embd).transpose(1, 2) # (B, nh, T, C)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_head))
        att = att.masked_fill(self.bias[:,:,:T, :T] == 0 ,float('-inf'))  # why is the masking with negative infinite? 
        att = F.softmax(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, bs) --> (B, nh, T, bs)
        y = y.transpose(1, 2).contiguous().view(B,T,C)

        y = self.c_proj(y)

        return y 
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x 

class Block(nn.Module):
    """
    Typical Transformer Block (pre-norm structure)

        Residual 1 
            LayerNorm1
            Attention
        Input2 = Input1 + Residual 1 

        Residual 2 
            LayerNorm2
            MLP 
        Output = Input2 + Residual 2
    """ 

    def __init__(self, config): 
        super(Block, self).__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x): 
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x 





class GPT2Configuration: 
    block_size = 1024 # context length 
    vocab_size = 50257 # vocab size of the byte-pair encodings 
    n_layers = 12
    n_head = 12
    n_embd = 768

class GPT2(nn.Module): 
    """
    GPT2 model initialization class. To use weights from the OpenAI checkpoint, naming must be similar 
    
    Using a learnable look-up table 
        Token Embeddings 
        Positional Embeddings 
        Transformer Blocks x Number of layers
    

    """
    def __init__(self, config): 
        super(GPT2, self).__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
        wpe =  nn.Embedding(self.config.vocab_size, self.config.n_embd),
        wte =  nn.Embedding(self.config.block_size, self.config.n_embd), 
        h = nn.ModuleList(Block(config) for _ in range(config.n_layers)),
        ln_f = nn.LayerNorm(config.n_embd)))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        # h = nn.ModuleList()
        
    def forward(self, idx): 
        B, T = idx.size()
        assert T <= self.config.block_size, f'Input sequence longer than max context size' 
        pos = torch.arange(0, T, dtype= torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        
        x = pos_emb + tok_emb
        for block in self.transformer.h: 
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.transformer(x)

        return logits