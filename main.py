from model import SamarthGPT2, GPT2Configuration
import tiktoken
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import GPT2LiteDataset

def train(): 
    config = GPT2Configuration
    model = SamarthGPT2(config)
    train_data = GPT2LiteDataset(B = 4, T = 32)

    # hyper-parameters
    optimizer = AdamW(model.parameters(), lr= 3e-4) # initalized from Karpathy implementation
    for i in range(50): 
        x, y = train_data.__next_batch__()
        print(x.shape, y.shape)

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