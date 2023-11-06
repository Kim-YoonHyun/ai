import sys
import os
import torch
import pandas as pd
from torch.utils.data import Dataset


pd.set_option('mode.chained_assignment',  None)


class AnomalyDataset(Dataset):
    """Anomaly detection dataset. """
    def __init__(self, tar_df_list, cond_df_list, raw_speed_df_list=None):
        self.tar_df_list = tar_df_list
        self.cond_df_list = cond_df_list
        self.raw_speed_df_list = raw_speed_df_list

    def __len__(self):
        return len(self.tar_df_list)


    def __getitem__(self, idx):
        tar_df = self.tar_df_list[idx]
        cond_df = self.cond_df_list[idx]
        if self.raw_speed_df_list:
            raw_speed_df = self.raw_speed_df_list[idx]
            speed_tensor = raw_speed_df.to_numpy()
        else:
            speed_tensor = torch.empty(1)

        x = tar_df.to_numpy()
        x_mark = cond_df.to_numpy()
        
        return x, x_mark, speed_tensor