import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch


# 한번에 모든 데이터를 올리고 사용하는 iterator
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
            b_label = b_label.to(self.device, dtype=torch.float32)
            
            del x
            return pred, b_label
        else:
            raise StopIteration


# 한번에 하나의 배치 데이터를 올리고 사용하는 iterator
class Iterator():
    def __init__(self, dataloader, model, device):
        self.dataloader = dataloader
        self.model = model
        self.device = device
            
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        x, b_label = next(iter(self.dataloader))
        
        x = x.to(self.device, dtype=torch.int)
        b_label = b_label.to(self.device).long()
        pred = self.model(x, b_label)
        
        del x

        return pred, b_label