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
from transformers import BertTokenizer



def subsequent_mask(size):
    ones_mat = np.ones((size, size))
    subsequent_mask = np.triu(ones_mat, k=1).astype('uint8')
    subsequent_mask = subsequent_mask == 0
    return subsequent_mask


class InputDataset(Dataset):
    def __init__(self, dataset_path, data_name_list):
        self.dataset_path = dataset_path
        self.data_name_list = data_name_list
        
    def __len__(self):
        return len(self.data_name_list)
    
    def __getitem__(self, idx):
        df = pd.read_csv(f'{self.dataset_path}/{self.data_name_list[idx]}', encoding='utf-8-sig')
        x = df['input'].values
        x_mask = np.where(x == 0, False, True)
        
        label = df['label'].values
        sub_mask = subsequent_mask(len(label))
        label_mask = np.where(label == 0, False, True)
        label_mask = label_mask & sub_mask

        return x, x_mask, label, label_mask


def get_dataloader(dataset_path, data_name_list, batch_size, shuffle=False, 
                   drop_last=False, num_workers=1, pin_memory=True):
    
    dataset = InputDataset(
        dataset_path=dataset_path, 
        data_name_list=data_name_list
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
    
    

