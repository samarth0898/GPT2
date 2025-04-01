import torch 
import tiktoken 
import numpy as np 
import os, glob

from torch.utils.data import Dataset

"""
Inputs and labels are the same size. 
    - To create both the input and labels, load [B X (T + 1)] tokens 
    - Use [B X (T-1)] as input 
    - Use [B X (T+1)] as output

"""

def dummy_data(file_path = 'input.txt'): 
    with open(file_path, 'r') as f:
        text = f.read() 

    text = text[:1000] # first 1000 words
    print(text)


class GPT2LiteDataset(): 
    """
    1 Million tokens in the shakesphere dataset 
    """
    
    def __init__(self, B, T, split): 
        super(GPT2LiteDataset, self).__init__ 
        with open('input.txt', 'r') as f: 
            text = f.read()
        encoder = tiktoken.get_encoding('gpt2')
        bpe_tokens = encoder.encode(text)
        self.bpe_tensor = torch.tensor(bpe_tokens)
        
        self.B, self.T = B, T
        self.current_pos = 0 


    def __next_batch__(self): 
        buf = self.bpe_tensor[self.current_pos: self.current_pos + self.B*self.T + 1]
        input = buf[:-1].view(self.B, self.T)
        output = buf[1:].view(self.B, self.T)
        self.current_pos = self.current_pos + (self.B*self.T)
        
        return input, output
    
    def __getitem__():
        pass

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
    
class GPT2Dataset():
    def __init__(self, B, T, split, data_root):
        self.B = B
        self.T = T
        self.process_rank = None
        self.num_processes = None
        assert split in {'train', 'val'}

        data_set = "edu_fineweb10B"
        shards = os.path.join(data_root, data_set)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        #     print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def __len__(self):
        pass
    def __getitem__(self):
        pass

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def __next_batch__(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


if __name__ == "__main__":
    dummy_data() 