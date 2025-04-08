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
        self.input_ids_list = []
        self.attention_mask_list = []
        self.token_type_ids_list = []
        self.label_ids_list = []
        self.batch_idx_list = []
        # self.x_list = []
        # self.b_label_list = []
        for batch_idx, batch_data in enumerate(dataloader):
            self.batch_idx_list.append(batch_idx)
            self.input_ids_list.append(batch_data[0])
            self.attention_mask_list.append(batch_data[1])
            self.token_type_ids_list.append(batch_data[2])
            self.label_ids_list.append(batch_data[3])
            
    def __len__(self):
        return len(self.batch_idx_list)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.batch_idx_list) - 1:
            self.count += 1
            
            # input ids
            input_ids = self.input_ids_list[self.count]
            input_ids = input_ids.to(self.device)
            
            # attention_mask
            attention_mask = self.attention_mask_list[self.count]
            attention_mask = attention_mask.to(self.device)
            
            # token type ids
            token_type_ids = self.token_type_ids_list[self.count]
            token_type_ids = token_type_ids.to(self.device)
            
            # labels
            labels = self.label_ids_list[self.count]
            labels = labels.to(self.device)
            
            # input 생성
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            inputs['token_type_ids'] = token_type_ids
            
            pred = self.model(**inputs)
            _, logit = pred[:2]
            b_label = labels
            del input_ids
            return pred, logit, b_label
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