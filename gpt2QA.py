import os 
import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 
from torch.nn.parameter import Parameter 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from torch.utils.data import Dataset, DistributedSampler, DataLoader 

# Custom trained modules 
from model.original_model import GPT, GPTConfig
from my_huggingface.optimization import GPT2Adam 
from my_huggingface.dataset_processor import preprocess_training_examples 
import tiktoken 
from datasets import load_dataset 
import warnings 
warnings.filterwarnings('ignore') 

# Importing a preconstructed model for testing 
from gpt2sqa.file_utils import PYTORCH_PRETRAINED_GPT2_CACHE, WEIGHTS_NAME, CONFIG_NAME 
from gpt2sqa.modeling_gpt2 import GPT2ModelForQuestionAnswering 

class GPT2forQA(nn.Module): 
    def _init_(self, pretrained_path): 
        super(GPT2forQA, self)._init_() 
        config = GPTConfig(vocab_size = 50304) 
        print(f"==> Config {config}") 
        self.gpt2 = GPT(config) 
        self.load_pretrained(pretrained_path) 

        # The idea is to predict the start and stop of the answer in the context 
        self.qa_output = nn.Linear(config.n_embd, 2) 
        
        # to predict the start and stop at the output 
    
    def load_pretrained(self, pretrained_path): 
        model_weights = torch.load(pretrained_path, weights_only=False)['model'] 
        self.gpt2.load_state_dict(model_weights) 
        print("==> Pretrained GPT2 loaded") 
    
    def forward(self, input_ids, token_type_ids = None, attention_mask= None, start_positions = None, end_positions = None): 
        sequence_output, loss = self.gpt2(input_ids) 
        logits = self.qa_output(sequence_output) 
        start_logits, end_logits = logits.split(1, dim = 2) 

def fine_tune(args, device = 'cuda'): 
    if torch.cude.is_available():
        device = 'cuda'
    # if torch.bacend
    # dataset 
    train_dataset = load_dataset("squad", split="train") 
    # every datapoint has ['id', 'title', 'context', 'question', 'answers'] 
    tokenizer = tiktoken.get_encoding('gpt2') 
    model = GPT2ModelForQuestionAnswering.from_pretrained(cache_dir=os.path.join(str(PYTORCH_PRETRAINED_GPT2_CACHE), 'distributed_{}'.format(args.local_rank))) 
    
    # training parameters 
    num_train_optimization_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs 
    model = model.to(device) 
    param_optimizer = list(model.named_parameters()) 
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # check if there naming conventions match. What parameters to decay? 
    optimizer_grouped_parameters = [ 
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, 
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ] 
    
    if num_train_optimization_steps is None: 
        num_train_optimization_steps = 1 
    
    optimizer = GPT2Adam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=num_train_optimization_steps) 
    # train_features = preprocess_training_examples(args, train_dataset = train_dataset, tokenizer= GPT2Tokenizer) 
    train_dataset = train_dataset.map( preprocess_training_examples, batched=True, remove_columns = train_dataset.column_names) 
    print(f"data tokenization completed") 
    
    if args.ddp: 
        sampler = DistributedSampler(train_dataset) 
    else: 
        sampler = None 
    
    train_loader = DataLoader(train_dataset, batch_size = 4) 
    global_step = 0 
    
    for _ in range(args.num_train_epochs): 
        epoch_loss = 0 
        grad_step_loss = 0 
        for step, batch in enumerate(train_loader): 
            input_ids, input_mask, start_positions, end_positions = batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions'] 
            # shift all tensors to CUDA 
            input_ids = torch.stack(input_ids, axis = -1).to(device)
            input_mask = torch.stack(input_mask, axis = -1).to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            loss = model(input_ids, None, input_mask, start_positions, end_positions) 
            
            if args.gradient_accumulation_steps > 1: 
                loss = loss / args.gradient_accumulation_steps 
            
            epoch_loss += loss.item() 
            grad_step_loss += loss.item()
            loss.backward() 
            
            if (step + 1) % args.gradient_accumulation_steps == 0: 
                optimizer.step() 
                optimizer.zero_grad() 
                global_step += 1 
                print(f'step {global_step} | train_loss {grad_step_loss}') 
                grad_step_loss = 0 
                

if __name__ == "_main_": 
    import argparse 

    parser = argparse.ArgumentParser() 

    parser.add_argument("--max_seq_length", default=1000, type=int, help="The maximum total input "
    "sequence length after WordPiece tokenization. Sequences " "longer than this will be truncated, and"
    " sequences shorter than this will be padded.") 

    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks, how much stride to take between chunks.") 
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum number of tokens for the question. Questions longer than this will " "be truncated to this length.") 
    args = parser.parse_args() 
    args.pre_path = r'/home/s.thopaiah/Documents/extras/GPT2/checkpoint/model_19072.pt' 
    
    args.train_batch_size = 16 
    args.gradient_accumulation_steps = 1 
    args.num_train_epochs = 3 
    args.learning_rate = 5e-5 
    args.warmup_proportion = 0.1 
    args.local_rank = -1 
    fine_tune_process = True 
    args.ddp = False 

    if fine_tune_process: 
        fine_tune(args)