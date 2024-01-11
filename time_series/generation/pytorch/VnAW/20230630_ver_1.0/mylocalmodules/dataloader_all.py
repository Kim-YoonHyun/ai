import sys
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def subsequent_mask(size):
    ones_mat = np.ones((size, size))
    subsequent_mask = np.triu(ones_mat, k=1).astype('uint8')
    subsequent_mask = subsequent_mask == 0
    return subsequent_mask


class InputDataset(Dataset):
    def __init__(self, input_list, label_list):
        self.input_ary = np.array(input_list)
        self.label_ary = np.array(label_list)
        
        input_mask_list = []
        label_mask_list = []
        for input_data, label_data in zip(tqdm(self.input_ary), self.label_ary):
            input_mask = np.where(input_data == 0, False, True)
            input_mask_list.append(input_mask.tolist())
            sub_mask = subsequent_mask(label_data.shape[-1])
            target_mask = np.where(label_data == 0, False, True)
            target_mask = target_mask & sub_mask
            label_mask_list.append(target_mask.tolist())
        self.input_mask_ary = np.array(input_mask_list)
        self.label_mask_ary = np.array(label_mask_list)
        
        
    def __len__(self):
        return len(self.input_ary)
    
    
    def __getitem__(self, idx):
        x = self.input_ary[idx]
        x_mask = self.input_mask_ary[idx]
        label = self.label_ary[idx]
        label_mask = self.label_mask_ary[idx]
        return x, x_mask, label, label_mask
    

def get_dataloader(input_list, label_list, batch_size, shuffle=False, 
                   drop_last=False, num_workers=1, pin_memory=True):
    
    dataset = InputDataset(
        input_list=input_list,
        label_list=label_list
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader

    
    

