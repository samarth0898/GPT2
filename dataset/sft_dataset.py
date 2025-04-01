"""
SFT dataset
Anthropic HH-RLHF dataset for continued autoregressive style training 

"""
import json, random
import tiktoken
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class TiktokenTokenizer():

    def __init__(self, name) -> None:
        self.enc = tiktoken.get_encoding(name)
        self.encode = lambda s: self.enc.encode(
            s, allowed_special={"<|endoftext|>"})
        self.pad_token = self.enc.eot_token

    def __call__(self,
                 text,
                 max_length=None,
                 padding=None,
                 truncation=False,
                 return_tensors=None):
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            mask += [0] * (max_length - len(ids))
            ids += [self.pad_token] * (max_length - len(ids))

        if return_tensors == "pt":
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask)

        return {"input_ids": ids, "attention_mask": mask}
    
class AnthropicHH(Dataset):
    def __init__(self, 
                 block_size, 
                 split = 'train', 
                 max_examples = None,
                 tokenizer_name = 'tiktoken/gpt2'):
        
        super(AnthropicHH, self).__init__()

        self.block_size = block_size 
        self.split = split 
        dataset = load_dataset("Anthropic/hh-rlhf")[self.split]['chosen']
        # import code; code.interact(local = locals())
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "tiktoken/gpt2":
            tokenizer =  TiktokenTokenizer('gpt2')

        cnt = 0
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")
    
    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y

AnthropicHH(block_size = 1024)
    
