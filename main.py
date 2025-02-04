from model import SamarthGPT2, GPT2Configuration
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
total_batch_size = 524288 # from GPT2 params
B = 4
T = 1024
grad_accum_steps = total_batch_size //(B*T)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'
print(f'selected device {device}')

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
    if device == 'cuda':
        model = torch.compile(model) # avoids the HBM <--> cache transfers, optimize memory transfer with kernel fusion
    else: 
        pass

    train_data = GPT2LiteDataset(B = B, T = T)

    # hyper-parameters
    # optimizer = AdamW(model.parameters(), lr= 3e-4, betas=(0.9, 0.95), eps = 10e-8) # initalized from Karpathy implementation
  
    optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device_type = device)
    
    torch.set_float32_matmul_precision('high') # TF32 
    
    for step in range(max_step): 
        time0 = time.time()
        # ADD IN THE VALIDATION HERE 

        model.train()
        optimizer.zero_grad()
        for mini_step in range(grad_accum_steps): # gradient accumulations to simulate true batch size 
            x, y = train_data.__next_batch__()
            x, y = x.to(), y.to()
            if device == 'cuda':
                with torch.autocast(device_type = device, dtype = torch.bfloat16): # automatic mixed precision training
                    logits, loss = model(x, labels = y)
            else: 
                logits, loss = model(x, labels = y)
            loss = loss / grad_accum_steps # manually averaging the losses
            loss.backward() # this automaticall accumulates the back-prop gradients
            break
        # clip the gradient norm according to GPT3 paper - preventing the model from getting very big alterations in the backprop
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) 
        
        # set the learning rate based on the step
        lr = get_lr(step)
        for param in optimizer.param_groups: 
            param['lr'] = lr 
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize() # for timing the code 
        if device == 'mps': 
            torch.mps.synchronize() # for debugging on mac
        time1 = time.time()
     
        # calculate tokens per second 
        dt = time1 - time0
        tokens_processed = train_data.B * train_data.T * grad_accum_steps #* ddp_world_size
        tokens_per_sec = tokens_processed / dt

        print(f"{step} -- train loss {loss.item():%04d} | norm {norm:%04d} | get_lr {lr} | token/sec {tokens_per_sec} | time {dt}")
      

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