import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment',  None)
from torch.utils.data import DataLoader, Dataset, SequentialSampler


class InputDataset(Dataset):
    def __init__(self, mode, input1, input2, input3, input4):
        self.mode = mode
        
        # train variables        
        self.input1 = input1
        self.input2 = input2
        
        # test variables
        self.input3 = input3
        self.input4 = input4

    def __len__(self):
        if self.mode == 'train':
            return len(self.input1)
        if self.mode == 'test':
            return len(self.input3)
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            x = self.input1[idx]
            y = self.input2[idx]
        if self.mode == 'test':
            x = self.input3[idx]
            y = self.input4[idx]
        
        return x, y


def get_dataloader(mode, input1, input2, input3, input4,
                   batch_size, 
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'):
    
    dataset = InputDataset(
        mode=mode,
        input1=input1,
        input2=input2,
        input3=input3,
        input4=input4
    )
    
    if sampler_name == 'SequentialSampler':
        sampler = SequentialSampler(dataset)
    else:
        pass
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        sampler=sampler
    )
    return dataloader



