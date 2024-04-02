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


def get_self_mask(x, pred_len, n_heads):
    # self-mask
    self_mask = np.where(x == -100, True, False)
    self_mask = np.tile(self_mask, (pred_len, 1))
    
    # look-ahead mask
    la_mask = np.triu(np.ones([pred_len, pred_len]), k=1)
    la_mask = np.where(la_mask==1, True, False)
    la_mask[self_mask & ~la_mask] = True
    
    # n_heads 갯수 만큼 늘리기
    self_mask = np.tile(self_mask, (n_heads, 1, 1))
    la_mask = np.tile(la_mask, (n_heads, 1, 1))
    
    return self_mask, la_mask
    

def fixed_max_norm(df, norm):
    col_list = df.columns.to_list()
    for col in col_list:
        num = norm[col]
        bos_num = df[col].values[0]
        df[col] = df[col].apply(lambda x: x/num)
        df[col].iloc[0] = bos_num
    return df
    
    
class VnawDataset(Dataset):
    def __init__(self, dataset_path, x_name_list, y_name_list, mark_name_list, scale, norm, n_heads):
        self.dataset_path = dataset_path
        self.x_name_list = x_name_list
        self.y_name_list = y_name_list
        self.mark_name_list = mark_name_list
        self.scale = scale
        self.norm = norm
        self.n_heads = n_heads
        # self.x_list = x_list
        # self.y_list = y_list
        # self.x_mark_list = x_mark_list
        # self.y_mark_list = y_mark_list
        # self.self_mask_list = self_mask_list
        # self.la_mask_list = la_mask_list

    def __len__(self):
        return len(self.x_name_list)
        
    def __getitem__(self, idx):
        x_name = self.x_name_list[idx]
        y_name = self.y_name_list[idx]
        mark_name = self.mark_name_list[idx]

        # df
        x_df = pd.read_csv(f'{self.dataset_path}/x/{x_name}')
        y_df = pd.read_csv(f'{self.dataset_path}/y/{y_name}')
        mark_df = pd.read_csv(f'{self.dataset_path}/mark/{mark_name}')
        
        # normalization
        if self.scale == 'fixed-max':
            x_df = fixed_max_norm(x_df, self.norm)
            y_df = fixed_max_norm(y_df, self.norm)
        
        # ary
        x = x_df.to_numpy()
        y = y_df.to_numpy()
        mark = mark_df.to_numpy()
        
        # mask
        x_ = np.squeeze(x[:, :1])
        self_mask, _ = get_self_mask(x_, len(x_), self.n_heads)
        
        y_ = np.squeeze(y[:, :1])
        _, la_mask = get_self_mask(y_, len(y_), self.n_heads)
        
        return x, mark, self_mask, y, mark, la_mask


class TestDataset(Dataset):
    def __init__(self, x_list, y_list, true_list, n_heads):
        self.x_list = x_list
        self.y_list = y_list
        self.true_list = true_list
        self.n_heads = n_heads

    def __len__(self):
        return len(self.x_list)
        
    def __getitem__(self, idx):
        x = self.x_list[idx]
        y = self.y_list[idx]
        true = self.true_list[idx]
        
        # mask
        x_ = np.squeeze(x[:, :1])
        self_mask, _ = get_self_mask(x_, len(x_), self.n_heads)
        
        y_ = np.squeeze(true[:, :1])
        _, la_mask = get_self_mask(y_, len(y_), self.n_heads)
        
        return x, _, self_mask, y, _, la_mask


def get_dataloader(dataset_path, 
                   x_name_list, 
                   y_name_list, 
                   mark_name_list, 
                   scale, norm,
                   n_heads,
                   batch_size, 
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'):
    
    dataset = VnawDataset(
        dataset_path=dataset_path, 
        x_name_list=x_name_list, 
        y_name_list=y_name_list, 
        mark_name_list=mark_name_list, 
        scale=scale,
        norm=norm,
        n_heads=n_heads
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


def get_test_dataloader(x_list, 
                        y_list, 
                        true_list,
                        n_heads,
                        batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                        num_workers=1, 
                        pin_memory=True,
                        sampler_name='SequentialSampler'):
    
    dataset = TestDataset(
        x_list=x_list, 
        y_list=y_list, 
        true_list=true_list,
        n_heads=n_heads
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

