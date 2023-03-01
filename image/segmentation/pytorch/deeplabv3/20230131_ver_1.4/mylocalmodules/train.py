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
        batch_idx_list = []
        x_list = []
        b_label_list = []
        for batch_idx, (x, b_label) in enumerate(dataloader):
            batch_idx_list.append(batch_idx)
            x_list.append(x)
            b_label_list.append(b_label)
        self.model = model
        self.device = device
        self.batch_idx_list = batch_idx_list
        self.x_list = x_list
        self.b_label_list = b_label_list
        self.count = -1
            
    def __len__(self):
        return len(self.batch_idx_list)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.batch_idx_list) - 1:
            self.count += 1
            x = self.x_list[self.count]
            x = x.to(self.device, dtype=torch.float)
            pred = self.model(x)
            
            pred = pred['out']
            b_label = self.b_label_list[self.count]
            return pred, b_label
        else:
            raise StopIteration



    

