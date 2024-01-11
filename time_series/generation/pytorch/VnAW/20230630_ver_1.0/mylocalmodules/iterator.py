import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
torch.cuda.empty_cache()


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
#         self.b_label_mask_list = []
#         for batch_idx, (x, x_mask, b_label, label_mask) in enumerate(dataloader):
#             self.batch_idx_list.append(batch_idx)
#             self.x_list.append(x)
#             self.b_label_list.append(b_label)
#             self.x_mask_list.append(x_mask)
#             self.b_label_mask_list.append(label_mask)
        
            
#     def __len__(self):
#         return len(self.dataloader)


#     def __iter__(self):
#         return self

    
#     def __next__(self):
#         if self.count < len(self.batch_idx_list) - 1:
#             self.count += 1
            
#             # x
#             x = self.x_list[self.count]
#             x = x.to(self.device, dtype=torch.int)
            
#             # x mask
#             x_mask = self.x_mask_list[self.count]
#             while len(x_mask.size()) < 4:
#                 x_mask = x_mask.unsqueeze(1)
#             x_mask = x_mask.to(self.device)
            
#             # b label
#             b_label = self.b_label_list[self.count]
#             b_label = b_label.to(self.device).long()
            
#             # b label mask
#             b_label_mask = self.b_label_mask_list[self.count]
#             while len(b_label_mask.size()) < 4:
#                 b_label_mask = b_label_mask.unsqueeze(1)
#             b_label_mask = b_label_mask.to(self.device)
            
#             # output
#             pred = self.model(
#                 input=x,
#                 target=b_label,
#                 input_mask=x_mask,
#                 target_mask=b_label_mask
#             )
#             del x, x_mask, b_label_mask
#             return pred, b_label
#         else:
#             raise StopIteration


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
        # ==
        x, x_mask, b_label, b_label_mask = next(iter(self.dataloader))
        x = x.to(self.device, dtype=torch.int)
        
        while len(x_mask.size()) < 4:
            x_mask = x_mask.unsqueeze(1)
        x_mask = x_mask.to(self.device)
        
        b_label = b_label.to(self.device).long()
        
        while len(b_label_mask.size()) < 4:
            b_label_mask = b_label_mask.unsqueeze(1)
        b_label_mask = b_label_mask.to(self.device)
        
        pred = self.model(
            input=x,
            target=b_label,
            input_mask=x_mask,
            target_mask=b_label_mask
        )
        
        del x, x_mask, b_label_mask

        return pred, b_label
