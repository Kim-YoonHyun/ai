import sys
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import SequentialSampler


pd.set_option('mode.chained_assignment',  None)


def get_self_mask(data, n_heads, pad_idx):
    self_mask = (data.cpu() == pad_idx).unsqueeze(1).unsqueeze(2) # 개11단sys.
    self_mask = self_mask.repeat(1, n_heads, data.shape[1], 1) # 개헤단단
    
    la_mask = torch.triu(torch.ones(data.shape[0], n_heads, data.shape[1], data.shape[1]), diagonal=1).bool()
    la_mask = self_mask | la_mask
    
    # # self-mask
    # self_mask = np.where(data==pad_idx, True, False)
    # self_mask = np.tile(self_mask, (len(data), 1))
    
    # look-ahead mask
    # la_mask = np.triu(np.ones([len(data), len(data)]), k=1)
    # la_mask = np.where(la_mask==1, True, False)
    # la_mask[self_mask & ~la_mask] = True
    
    # n_heads 갯수 만큼 늘리기
    # self_mask = np.tile(self_mask, (n_heads, 1, 1))
    # la_mask = np.tile(la_mask, (n_heads, 1, 1))
    
    return self_mask, la_mask


def get_enc_dec_mask(enc_data, dec_data, n_heads, pad_idx):
    enc_dec_mask = (enc_data == pad_idx).unsqueeze(1).unsqueeze(2)
    enc_dec_mask = enc_dec_mask.repeat(1, n_heads, dec_data.shape[1], 1)
    
    # enc_dec_mask = np.expand_dims((enc_data == pad_idx), axis=0)
    # enc_dec_mask = np.expand_dims(enc_dec_mask, axis=0)
    # enc_dec_mask = np.tile(enc_dec_mask, (n_heads, len(dec_data), 1))
    
    # enc_dec_mask 의 shape
    # (head num, y length, x length)
    
    return enc_dec_mask
    

class TextDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

        
    def __len__(self):
        return len(self.x_list)
    
    
    def __getitem__(self, idx):
        # ary
        x = np.array(self.x_list[idx])
        y = np.array(self.y_list[idx])
        return x, y#, x_self_mask, y_la_mask, enc_dec_mask


def get_dataloader(dataset, batch_size):
    
    # sampler = SequentialSampler(dataset)
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # drop_last=False, 
        # num_workers=1, 
        # pin_memory=True,
        # sampler=sampler
    )
    return dataloader



