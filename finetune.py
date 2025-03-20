""""
fine-tuning strategies on a GPT2 to check if alignment from human feedback help older, smaller models like GPT2

Alignment stages (similar to InstructGPT)
    1. SFT 
    2. Reward Model 
    3. RLHF

Some capbilites from hugging face are used here (SFTTrainer etc. )
"""
import torch 
from GPT2.model.custom_model import GPT2Configuration, SamarthGPT2

def train_sft():
    pass 
def train_reward_model():
    pass 
def train_rlhf(): 
    pass 

def fine_tune(ckpt_file): 
    # model_ckpt = torch.load(ckpt_file)
    gpt = SamarthGPT2(GPT2Configuration)


ckpt_file = ""
fine_tune(ckpt_file)