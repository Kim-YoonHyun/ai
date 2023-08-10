import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
        self.b_token_ids_list = []
        self.b_valid_length_list = []
        self.b_segment_ids_list = []
        self.b_label_list = []
        for batch_idx, (b_token_ids, b_valid_length, b_segment_ids, b_label) in enumerate(dataloader):
            self.batch_idx_list.append(batch_idx)
            self.b_token_ids_list.append(b_token_ids)
            self.b_valid_length_list.append(b_valid_length)
            self.b_segment_ids_list.append(b_segment_ids)
            self.b_label_list.append(b_label)
        
    def __len__(self):
        return len(self.batch_idx_list)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.batch_idx_list) - 1:
            self.count += 1
            
            # token ids
            b_token_ids = self.b_token_ids_list[self.count]
            b_token_ids = b_token_ids.to(self.device, dtype=torch.long)
            
            # valid length
            b_valid_length = self.b_valid_length_list[self.count]
            
            # segment ids
            b_segment_ids = self.b_segment_ids_list[self.count]
            b_segment_ids = b_segment_ids.to(self.device, dtype=torch.long)
            
            # pred
            pred = self.model(b_token_ids, b_valid_length, b_segment_ids)
            
            b_label = self.b_label_list[self.count]
            b_label = b_label.to(self.device, dtype=torch.long)
            
            del b_token_ids, b_valid_length, b_segment_ids
            return pred, b_label
        else:
            raise StopIteration


# # 한번에 하나의 배치 데이터를 올리고 사용하는 iterator
# class Iterator():
#     def __init__(self, dataloader, model, device):
#         self.dataloader = dataloader
#         self.model = model
#         self.device = device
            
#     def __len__(self):
#         return len(self.dataloader)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         x, b_label = next(iter(self.dataloader))
        
#         x = x.to(self.device, dtype=torch.int)
#         b_label = b_label.to(self.device).long()
#         pred = self.model(x, b_label)
        
#         del x

#         return pred, b_label