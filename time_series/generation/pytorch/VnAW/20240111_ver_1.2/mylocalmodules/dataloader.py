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


pd.set_option('mode.chained_assignment',  None)


class AnomalyDataset(Dataset):
    """Anomaly detection dataset. """
    def __init__(self, x_df_list, x_mark_df_list, y_df_list, y_mark_df_list, pred_len):
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

    def normalization(self, x):
        min_num = np.min(x)
        x = x - min_num
        max_num = np.max(x)
        x_norm = x / max_num
        x_norm = np.expand_dims(x_norm, axis=-1)
        return x_norm
    
    def __getitem__(self, idx):
        df = pd.read_csv(f'{self.dataset_path}/{self.data_name_list[idx]}', encoding='utf-8-sig')
        
        # x = df[['input_1', 'input_2']].values
        x = df['input'].values
        
        
        # # normalization
        # x_1 = df['input_1'].values
        # x_1_norm = self.normalization(x_1)
        # x_2 = df['input_2'].values
        # x_2_norm = self.normalization(x_2)
        # x = np.concatenate((x_1_norm, x_2_norm), axis=-1)
        
        # x mask
        mask_size = x.shape[0]
        x_for_mask = np.squeeze(x[:mask_size, :1])
        x_mask = np.where(x_for_mask == 0, False, True)
        
        
        label = df['label'].values
        # sub_mask = subsequent_mask(mask_size)
        # label_mask = np.where(label == 0, False, True)
        # label_mask = label_mask[:mask_size]
        # label_mask = label_mask & sub_mask
        label = np.expand_dims(label, axis=-1)
        return x, x_mask, label


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
    
    

