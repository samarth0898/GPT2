from model import SamarthGPT2, GPT2Configuration, configure_optimizer
import time, math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import GPT2LiteDataset

max_lr = 6e-4
min_lr = max_lr * 0.1
# step: 2 ** 19 tokens 
# total: 10e9

max_step = 19073 # total_steps = 10e9 / 2**19 ~ 19,073
warmup_steps = 715 # 375e6/2**19 ~ 715 


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps():
    device = 'mps'
print(f'selected device {device}')

total_batch_size = 524288 # from GPT2 params
B = 16
T = 1024

grad_accum_steps = total_batch_size //(B*T)

# cosine annealing learning rate scheduler with warmup 
def get_lr(iteration):
    if iteration <= warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps
    if iteration > max_step:
        return min_lr
    
    decay_ratio = (iteration - warmup_steps) / (max_step - warmup_steps)
    assert 0<= decay_ratio <= 1

    coeff  = 0.5 * (1.0 + math.pi + decay_ratio)
    return min_lr + coeff * (max_lr - min_lr)

def train(): 
    config = GPT2Configuration
    model = SamarthGPT2(config)
    model = torch.compile(model) # avoids the HBM <--> cache transfers, optimize memory transfer with kernel fusion

    train_data = GPT2LiteDataset(B = 4, T = 32)

    # hyper-parameters
    # optimizer = AdamW(model.parameters(), lr= 3e-4, betas=(0.9, 0.95), eps = 10e-8) # initalized from Karpathy implementation
    device = 'cuda'
    optimizer = configure_optimizer(weight_decay = 0.1, lr = 6e-4, device = device)
    
    torch.set_float32_matmul_precision('high') # TF32 
    for i in range(50): 
        time0 = time.time()
        optimizer.zero_grad()
        for mini_step in range(grad_accum_steps): # gradient accumulations 
            x, y = train_data.__next_batch__()
            x, y = x.to(), y.to()
            with torch.autocast(device_type = device, dtype = torch.bfloat16): # automatic mixed precision training
                logits, loss = model(x, labels = y)
            loss = loss / grad_accum_steps # manually averaging the losses
            loss.backward() # this automaticall accumulates the back-prop gradients

        # clip the gradient norm according to GPT3 paper - preventing the model from getting very big alterations in the backprop
        norm = torch.nn.utils.clip_grad_norm(model.parameter(), 1.0) 
        # set the learning rate for this 
        lr = get_lr(i)
        for param in optimizer.param_groups: 
            param['lr'] = lr 
        optimizer.step()
        
        torch.cuda.synchronize()
        time1 = time.time()
        # print(f'{(train_data.B * train_data.T)/ }')

        print(f"{i} -- loss {loss.item()} | norm {norm} | get_lr {lr} | time {time1 - time0}")
      

def infer(): 
    config = GPT2Configuration
    model = SamarthGPT2(config)
    
    # GPT BPE tokenizer ~ tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    token = encoder.encode("Hello, I'm Samarth")
    token = torch.tensor(token, dtype = torch.long)
    
    num_return_sequence = 5
    # test 5 iterations of the output 
    token = token.unsqueeze(0).repeat(num_return_sequence, 1)
    if torch.cuda.is_available():
        x = token.to('cuda')
    if torch.backends.mps.is_available():
        model = model.to('mps')
        x = token.to('mps')

    # iteratively infer 
    max_generation_length = 30
    while x.size(1) < max_generation_length: 
        with torch.no_grad():
            logits = model(x) # B, T , Vocab_size 
            
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)

            topk_probs, topk_idx = torch.topk(probs, 50, dim = -1) # top k choices from the vocabulary
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)
            x = torch.cat((x, xcol), dim  = 1)
    
    # decoding the probits 
    for i in range(num_return_sequence): 
        token = x[i,:max_generation_length].tolist()
        decoded = encoder.decode(token)
        print("==", decoded)

if __name__ == "__main__":
    #infer()
    train()