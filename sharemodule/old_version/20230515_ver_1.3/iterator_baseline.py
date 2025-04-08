import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
    
class Iterator():
    def __init__(self, dataloader, model, device):
        self.model = model
        self.device = device
        self.count = -1
        self.batch_idx_list = []
        self.x_list = []
        self.b_label_list = []
        for batch_idx, (x, b_label) in enumerate(dataloader):
            self.batch_idx_list.append(batch_idx)
            self.x_list.append(x)
            self.b_label_list.append(b_label)
        
            
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
            b_label = self.b_label_list[self.count]
            return pred, b_label
        else:
            raise StopIteration


