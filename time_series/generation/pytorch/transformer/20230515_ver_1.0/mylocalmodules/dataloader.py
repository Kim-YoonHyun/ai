import sys
import os
# import joblib
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


pd.set_option('mode.chained_assignment',  None)


class AnomalyDataset(Dataset):
    """Anomaly detection dataset. """
    def __init__(self, mode, tar_df_list, cond_df_list):
        self.mode__ = mode
        self.tar_df_list = tar_df_list
        self.cond_df_list = cond_df_list

    def __len__(self):
        if self.mode__ == 'train':
            return len(self.tar_df_list)
        if self.mode__ == 'test':
            return len(self.tar_df_list)


    def __getitem__(self, idx):
        tar_df = self.tar_df_list[idx]
        cond_df = self.cond_df_list[idx]

        x = tar_df.to_numpy()
        x_mark = cond_df.to_numpy()
        
        return x, x_mark