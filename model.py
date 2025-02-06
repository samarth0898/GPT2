import torch.nn.functional as F
import torch.nn as nn 
import torch
import math
import inspect

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

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, 
                                                                                                       config.block_size))
        self.flash_attn = True 

    def forward(self, x):
        B, T, C = x.size() 

        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd,  dim = 2)

        if not self.flash_attn:
            k = k.view(B, T , self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C // nh)
            q = q.view(B, T , self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C // nh)
            v = v.view(B, T , self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C // nh)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_head)) # (B, nh, T, T)
            att = att.masked_fill(self.bias[:,:,:T, :T] == 0 ,float('-inf'))  # why is the masking with negative infinite? 
            att = F.softmax(att, dim = -1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, C // nh ) --> (B, nh, T, C // nh)
        
        if self.flash_attn: 
            y = F.scaled_dot_product_attention(q, k, v , is_causal = True)
        y = y.transpose(1, 2).contiguous().view(B,T,C) # (B, T, C)

        y = self.c_proj(y)

        return y 
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.RESIDUAL_INIT = 1

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x 



class GPT2Configuration: 
    
    block_size = 1024 # context length 
    vocab_size = 50304 # vocab size of the byte-pair encodings 
    n_layers = 12
    n_head = 12
    n_embd = 768

class SamarthGPT2(nn.Module): 
    """
    GPT2 model initialization class. To use weights from the OpenAI checkpoint, naming must be similar 
    
    Using a learnable look-up table 
        Token Embeddings 
        Positional Embeddings 
        Transformer Blocks x Number of layers
    

    """
    def __init__(self, config): 
        super(SamarthGPT2, self).__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
        wpe =  nn.Embedding(self.config.vocab_size, self.config.n_embd),
        wte =  nn.Embedding(self.config.block_size, self.config.n_embd),  # this layer is the same as lm_head layer,  weight tying from 'attention is all you need' 
        h = nn.ModuleList(Block(config) for _ in range(config.n_layers)),
        ln_f = nn.LayerNorm(config.n_embd)))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        # h = nn.ModuleList()

        # weight sharing/ weight tying scheme between wte(forward) and lm_head(inverse)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init__weights__)

    def __init__weights__(self, module): 
        
        if isinstance(module, nn.Linear): 
            std = 0.02
            if hasattr(module, 'RESIDUAL_INIT'): 
                std *= (2 * self.config.n_layers) ** -0.5            # adjust the variance of the init with the number of residual connections, no variance accumulation 
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, labels = None): 
        B, T = idx.size()
        assert T <= self.config.block_size, f'Input sequence longer than max context size' 
        pos = torch.arange(0, T, dtype= torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        
        x = pos_emb + tok_emb
        # print(f'after position embeddings {pos_emb.shape}, {tok_emb.shape}, {x.shape}')
        for block in self.transformer.h: 
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if labels is not None: 
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if device_type == "cuda":
            use_fused = fused_available and device_type == "cuda"
        # if master_process:
        else: 
            use_fused = False
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer