import sys
import os
import pandas as pd
import json
import copy
import time
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')



class Iterator():
    def __init__(self, dataloader, model, device):
        self.model = model
        self.device = device

        self.batch_idx_list = []
        self.x_list = []
        self.x_mark_list = []
        self.y_list = []
        self.y_mark_list = []
        self.s_mask_list = []
        self.la_mask_list = []
        
        for batch_idx, (x, x_mark, s_mask, y, y_mark, la_mask) in enumerate(tqdm(dataloader)):
            self.batch_idx_list.append(batch_idx)
            self.x_list.append(x)
            self.x_mark_list.append(x_mark)
            self.y_list.append(y)
            self.y_mark_list.append(y_mark)
            self.s_mask_list.append(s_mask)
            self.la_mask_list.append(la_mask)
        # self.batch_idx_list = batch_idx_list
        # self.x_list = x_list
        # self.x_mark_list = x_mark_list
        # self.s_mask_list = s_mask_list
        
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
            
            x_mark = self.x_mark_list[self.count]
            x_mark = torch.tensor(x_mark, dtype=torch.float32)
            x_mark = x_mark.to(self.device)
            
            s_mask = self.s_mask_list[self.count]
            s_mask = torch.tensor(s_mask)
            s_mask = s_mask.to(self.device)
            
            y = self.y_list[self.count]
            y_input = y.to(self.device, dtype=torch.float)
            
            y_mark = self.y_mark_list[self.count]
            y_mark = torch.tensor(y_mark, dtype=torch.float32)
            y_mark = y_mark.to(self.device)
            
            la_mask = self.la_mask_list[self.count]
            la_mask = torch.tensor(la_mask)
            la_mask = la_mask.to(self.device)
            
            pred = self.model(
                x=x_input,
                x_mark=x_mark, 
                y=y_input, 
                y_mark=y_mark,
                enc_self_mask=s_mask,
                look_ahead_mask=la_mask,
                enc_dec_mask=s_mask
                )
            
            logit = pred
            b_label = y_input
            return pred, logit, b_label
        else:
            raise StopIteration



    





# # 전체입력
# class Iterator():
#     def __init__(self, dataloader, model, device):
#         self.dataloader = dataloader
#         self.model = model
#         self.device = device
#         self.count = -1
#         self.batch_idx_list = []
#         self.x_list = []
#         self.x_mask_list = []
#         self.b_label_list = []
#         for batch_idx, (x, x_mask, b_label) in enumerate(dataloader):
#             self.batch_idx_list.append(batch_idx)
#             self.x_list.append(x)
#             self.b_label_list.append(b_label)
#             self.x_mask_list.append(x_mask)
        
            
#     def __len__(self):
#         return len(self.dataloader)


#     def __iter__(self):
#         return self

    
#     def __next__(self):
#         if self.count < len(self.batch_idx_list) - 1:
#             self.count += 1
            
#             # x
#             x = self.x_list[self.count]
#             x = x.to(self.device, dtype=torch.float32)
            
#             # x mask
#             x_mask = self.x_mask_list[self.count]
#             while len(x_mask.size()) < 4:
#                 x_mask = x_mask.unsqueeze(1)
#             x_mask = x_mask.to(self.device)
            
#             # b label
#             b_label = self.b_label_list[self.count]
#             b_label = b_label.to(self.device, dtype=torch.float32)
#             # # b label mask
#             # b_label_mask = self.b_label_mask_list[self.count]
#             # while len(b_label_mask.size()) < 4:
#             #     b_label_mask = b_label_mask.unsqueeze(1)
#             # b_label_mask = b_label_mask.to(self.device)
            
#             # output
#             pred = self.model(
#                 input=x,
#                 input_mask=x_mask,
#             )
#             del x, x_mask
#             return pred, b_label
#         else:
#             raise StopIteration


