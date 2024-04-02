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


class AnomalyDataset(Dataset):
    """Anomaly detection dataset. """
    def __init__(self, x_df_list, x_mark_df_list, y_df_list, y_mark_df_list, n_heads, pred_len):
        self.x_df_list = x_df_list
        self.x_mark_df_list = x_mark_df_list
        self.y_df_list = y_df_list
        self.y_mark_df_list = y_mark_df_list
        self.n_heads = n_heads
        self.pred_len = pred_len


    def __len__(self):
        return len(self.x_df_list)
    
    
    def get_self_mask(self, x):
        # mask_len = x.shape[0]
        
        # self-mask
        self_mask = np.where(x == -100, True, False)
        self_mask = np.tile(self_mask, (self.pred_len, 1))
        
        # look-ahead mask
        la_mask = np.triu(np.ones([self.pred_len, self.pred_len]), k=1)
        la_mask = np.where(la_mask==1, True, False)
        la_mask[self_mask & ~la_mask] = True
        
        # n_heads 갯수 만큼 늘리기
        self_mask = np.tile(self_mask, (self.n_heads, 1, 1))
        la_mask = np.tile(la_mask, (self.n_heads, 1, 1))
        
        return self_mask, la_mask

        
    def __getitem__(self, idx):
        x_df = self.x_df_list[idx]
        y_df = self.y_df_list[idx]
        
        x_ = x_df.iloc[:, :1].values
        x_ = np.squeeze(x_)
        self_mask, _ = self.get_self_mask(x_)
        x_mark_df = self.x_mark_df_list[idx]
        
        y_ = y_df.iloc[:, :1].values
        y_ = np.squeeze(y_)
        _, look_ahead_mask = self.get_self_mask(y_)
        y_mark_df = self.y_mark_df_list[idx]
        
        x = x_df.to_numpy()
        x_mark = x_mark_df.to_numpy()
        y = y_df.to_numpy()
        y_mark = y_mark_df.to_numpy()

        return x, x_mark, self_mask, y, y_mark, look_ahead_mask



def get_dataloader(x_df_list, x_mark_df_list, y_df_list, y_mark_df_list, n_heads, pred_len,
                   batch_size, 
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'):
    
    dataset = AnomalyDataset(
        x_df_list=x_df_list,
        x_mark_df_list=x_mark_df_list,
        y_df_list=y_df_list,
        y_mark_df_list=y_mark_df_list,
        n_heads=n_heads, 
        pred_len=pred_len
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

