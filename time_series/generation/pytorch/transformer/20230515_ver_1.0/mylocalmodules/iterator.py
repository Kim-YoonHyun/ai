import sys
import os
import pandas as pd
import json
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')



class Iterator():
    def __init__(self, dataloader, model, device):
        self.model = model
        self.device = device

        batch_idx_list = []
        x_list = []
        x_mark_list = []
        for batch_idx, (x, x_mark, _) in enumerate(dataloader):
            batch_idx_list.append(batch_idx)
            x_list.append(x)
            x_mark_list.append(x_mark)
        self.batch_idx_list = batch_idx_list
        self.x_list = x_list
        self.x_mark_list = x_mark_list
        self.count = -1
            
    def __len__(self):
        return len(self.batch_idx_list)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.batch_idx_list) - 1:
            self.count += 1
            x = self.x_list[self.count]
            x_input = x.to(self.device, dtype=torch.float)
            x_input = torch.nan_to_num(x_input)
            
            x_mark = self.x_mark_list[self.count]
            x_mark = torch.tensor(x_mark, dtype=torch.float32)
            x_mark = x_mark.to(self.device)
            x_mark = torch.nan_to_num(x_mark)
            pred = self.model(x_input, x_mark, x_input, x_mark)
            
            return pred, x_input
        else:
            raise StopIteration



    

