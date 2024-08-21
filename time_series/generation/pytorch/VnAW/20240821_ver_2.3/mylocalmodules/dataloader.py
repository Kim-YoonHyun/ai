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


def get_enc_dec_mask(x_self_mask, y_self_mask):
    enc_dec_mask = y_self_mask[:, :, :x_self_mask.shape[1]]
    return enc_dec_mask
    

def fixed_max_norm(df, norm):
    col_list = df.columns.to_list()
    for col in col_list:
        if 'Unnamed' in col:
            continue
        num = norm[col]
        df[col] = df[col].apply(lambda x: x/num)
    return df
    
    
class VnawDataset(Dataset):
    def __init__(self, mode, n_heads, x_p,
                 dataset_path=None, 
                 x_name_list=None, y_name_list=None, mark_name_list=None, 
                 scale=None, norm=None, configuration=None
                 ):
        self.mode = mode
        self.n_heads = n_heads
        self.x_p = x_p
        
        # train variables        
        self.dataset_path = dataset_path
        self.x_name_list = x_name_list
        self.y_name_list = y_name_list
        self.mark_name_list = mark_name_list
        self.scale = scale
        self.norm = norm
        self.configuration = configuration
        
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
            
        if self.mode == 'test':
            x_len = len(x)
            if self.configuration == 'regression':
                y = np.zeros((x_len, 1))
            elif self.configuration == 'bos time series':
                y[-600:] = 0
            else:
                print('잘못된 configuration 입니다')
                sys.exit()
        
        # x percent
        if self.x_p > 0:
            p_len = int(len(x) * self.x_p/100)
            p_seg_x = x[-p_len:, :].T
            p_seg_x = np.expand_dims(np.concatenate(p_seg_x, axis=0), axis=-1)
            y = np.concatenate((p_seg_x, y), axis=0)
        
        # mask
        x_ = np.squeeze(x[:, :1])
        x_self_mask, _ = get_self_mask(x_, len(x_), self.n_heads)
        y_ = np.squeeze(y[:, :1])
        y_self_mask, y_la_mask = get_self_mask(y_, len(y_), self.n_heads)
        enc_dec_mask = get_enc_dec_mask(x_self_mask, y_self_mask)
        return x, mark, x_self_mask, y, mark, enc_dec_mask, y_la_mask


def get_dataloader(mode,
                   n_heads,
                   x_p,
                   batch_size, 
                   dataset_path=None, 
                   x_name_list=None, y_name_list=None, mark_name_list=None, 
                   scale=None, norm=None, configuration=None,
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'
                   ):
    
    dataset = VnawDataset(
        mode=mode,
        n_heads=n_heads,
        x_p=x_p,
        dataset_path=dataset_path, 
        x_name_list=x_name_list, 
        y_name_list=y_name_list, 
        mark_name_list=mark_name_list, 
        scale=scale,
        norm=norm,
        configuration=configuration
        # x_list=x_list,
        # y_list=y_list,
        # mark_list=mark_list
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



