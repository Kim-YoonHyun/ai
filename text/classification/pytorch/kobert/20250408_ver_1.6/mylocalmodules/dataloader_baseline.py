import sys
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
# import csv
# import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torch.autograd import Variable
from torch.utils.data import SequentialSampler


pd.set_option('mode.chained_assignment',  None)


def fixed_max_norm(df, norm):
    col_list = df.columns.to_list()
    for col in col_list:
        if 'Unnamed' in col:
            continue
        num = norm[col]
        df[col] = df[col].apply(lambda x: x/num)
    return df
    
    
class NetworkDataset(Dataset):
    def __init__(self, mode, 
                 dataset_path=None, x_name_list=None, y_name_list=None,
                 scale=None, norm=None, configuration=None
                 ):
        self.mode = mode
        
        # train variables        
        self.dataset_path = dataset_path
        self.x_name_list = x_name_list
        self.y_name_list = y_name_list
        self.scale = scale
        self.configuration = configuration
        
    def __len__(self):
        return len(self.x_name_list)
        
    def __getitem__(self, idx):
        # ======================
        # 경로를 통해 csv 를 불러오는 경우
        x_name = self.x_name_list[idx]
        y_name = self.y_name_list[idx]

        # df
        x_df = pd.read_csv(f'{self.dataset_path}/x/{x_name}')
        y_df = pd.read_csv(f'{self.dataset_path}/y/{y_name}')
        
        # normalization
        if self.scale == 'fixed-max':
            x_df = fixed_max_norm(x_df, self.norm)
            y_df = fixed_max_norm(y_df, self.norm)
        
        # ary
        x = x_df.to_numpy()
        y = y_df.to_numpy()
            
        # test 데이터셋인 경우 전처리 code
        if self.mode == 'test':
            if self.configuration == '':
                pass
            else:
                print('잘못된 configuration 입니다')
                sys.exit()
            
        return x, y


def get_dataloader(mode,
                   batch_size, 
                   dataset_path=None, 
                   x_name_list=None, 
                   y_name_list=None, #mark_name_list=None, 
                   scale=None, norm=None, 
                   configuration=None,
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'
                   ):
    
    dataset = NetworkDataset(
        mode=mode,
        dataset_path=dataset_path, 
        x_name_list=x_name_list, 
        y_name_list=y_name_list, 
        scale=scale,
        norm=norm,
        configuration=configuration
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



