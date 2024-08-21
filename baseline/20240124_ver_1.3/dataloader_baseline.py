import sys
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler


class InputDataset(Dataset):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        
    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        x = self.input1[idx]
        y = self.input2[idx]
        return x, y


def get_dataloader(input1, input2, 
                   batch_size, 
                   shuffle=False, drop_last=False, 
                   num_workers=1, pin_memory=True,
                   sampler_name='SequentialSampler'):
    
    dataset = InputDataset(
        input1=input1, 
        input2=input2
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
    
    

