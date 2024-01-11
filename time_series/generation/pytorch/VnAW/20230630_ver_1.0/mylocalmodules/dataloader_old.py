import sys
import os

import numpy as np
import csv
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
from transformers import BertTokenizer



def load_csv(file_path):
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        
        line_list = []
        
        for line in csv_reader:
            line[0] = line[0].replace(';', '')
            line_list.append(line)
    return line_list


def subsequent_mask(size):
    ones_mat = np.ones((1, size, size))
    subsequent_mask = np.triu(ones_mat, k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    return subsequent_mask


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length):
        pad_token_idx = tokenizer.pad_token_id
        csv_datas = load_csv(file_path)
        
        self.doc_list = []
        
        for line in tqdm(csv_datas):
            # input
            input = tokenizer.encode(line[0], max_length=max_length, truncation=True)
            rest = max_length - len(input)
            input = torch.tensor(input + [pad_token_idx]*rest)
            
            # target
            target = tokenizer.encode(line[1], max_length=max_length, truncation=True)
            rest = max_length - len(target)
            target = torch.tensor(target + [pad_token_idx]*rest)
            
            input_string = tokenizer.convert_ids_to_tokens(input)
            input_mask = (input != pad_token_idx).unsqueeze(-2)
            target_string = tokenizer.convert_ids_to_tokens(target)
            target_mask = self.make_std_mask(target, pad_token_idx)
            doc = {
                'input_str':input_string,
                'input':input,
                'input_mask': input_mask,
                'target_str':target_string,
                'target':target,
                'target_mask':target_mask,
                'token_num': (target[..., 1:] != pad_token_idx).data.sum()
            }
            self.doc_list.append(doc)
            
    
    @staticmethod
    def make_std_mask(target, pad_token_idx):
        target_mask = (target != pad_token_idx).unsqueeze(-2)
        target_mask = target_mask & Variable(subsequent_mask(target.size(-1)).type_as(target_mask.data))
        return target_mask.squeeze()
        

    def __len__(self):
        return len(self.doc_list)
    
    
    def __getitem__(self, idx):
        item = self.doc_list[idx]
        return item





