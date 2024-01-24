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
    def __init__(self, x_path, x_mark_path, y_path, y_mark_path, pred_len):
        
        # x_path = f'{dataset_path}/train/x'
        # x_mark_path = f'{dataset_path}/train/x_mark'
        # y_path = f'{dataset_path}/train/y'
        # y_mark_path = f'{dataset_path}/train/y_mark'
        
        x_name_list = os.listdir(x_path)
        x_name_list.sort()
        x_mark_name_list = os.listdir(x_mark_path)
        x_mark_name_list.sort()
        
        y_name_list = os.listdir(y_path)
        y_name_list.sort()
        y_mark_name_list = os.listdir(y_mark_path)
        y_mark_name_list.sort()
        
        # x
        x_df_list = []
        for x_n in x_name_list:
            x_df = pd.read_csv(f'{x_path}/{x_n}')
            x_df_list.append(x_df)
        x_mark_df_list = []
        for x_mn in x_mark_name_list:
            x_mark_df = pd.read_csv(f'{x_mark_path}/{x_mn}')
            x_mark_df_list.append(x_mark_df)
            
        # y
        y_df_list = []
        for y_n in y_name_list:
            y_df = pd.read_csv(f'{y_path}/{y_n}')
            y_df_list.append(y_df)
        y_mark_df_list = []
        for y_mn in y_mark_name_list:
            y_mark_df = pd.read_csv(f'{y_mark_path}/{y_mn}')
            y_mark_df_list.append(y_mark_df)
        
        self.x_df_list = x_df_list
        self.x_mark_df_list = x_mark_df_list
        self.y_df_list = y_df_list
        self.y_mark_df_list = y_mark_df_list
        self.pred_len = pred_len


    def __len__(self):
        return len(self.x_df_list)
    
    
    def get_self_mask(self, x):
        mask_len = x.shape[0]
        
        # self-mask
        self_mask = np.where(x == -100, True, False)
        self_mask = np.tile(self_mask, (mask_len, 1))
        
        # look-ahead mask
        la_mask = np.triu(np.ones([mask_len, mask_len]), k=1)
        la_mask = np.where(la_mask==1, True, False)
        la_mask[self_mask & ~la_mask] = True
        
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



def get_dataloader(x_path, x_mark_path, y_path, y_mark_path, 
                   pred_len, batch_size, 
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name=None):
    
    dataset = AnomalyDataset(
        x_path=x_path, 
        x_mark_path=x_mark_path, 
        y_path=y_path, 
        y_mark_path=y_mark_path,
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
    
    

