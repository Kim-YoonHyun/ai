import sys
import os
import pandas as pd
import numpy as np
import json
import copy
import time
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/kimyh/python/ai')
from sharemodule import classificationutils as cfum


def iterator(mode, purpose, dataloader, model, device, loss_function=None, optimizer=None, max_grad_norm=None):
    
    # loss, 결과 초기화
    logit_list = []
    b_label_list = []
    loss_sum = 0
    result_list = []

    # classification 결과 초기화
    if purpose == 'classification':
        pred_label_list = []
        pred_reliability_list = []
        pred_2nd_label_list = []
    
    iter_n = 0
    
    # 이터레이터 진행
    for batch_idx, (x, x_mark, s_mask, y, y_mark, la_mask) in enumerate(tqdm(dataloader)):
        x_input = x.to(device, dtype=torch.float)
        y_input = y.to(device, dtype=torch.float)
        
        x_mark = torch.tensor(x_mark, dtype=torch.float32)
        y_mark = torch.tensor(y_mark, dtype=torch.float32)
        x_mark = x_mark.to(device)
        y_mark = y_mark.to(device)
        
        s_mask = torch.tensor(s_mask)
        s_mask = s_mask.to(device)
        
        la_mask = torch.tensor(la_mask)
        la_mask = la_mask.to(device)
        
        pred = model(
            x=x_input,
            y=y_input, 
            x_mark=x_mark, 
            y_mark=y_mark,
            enc_self_mask=s_mask,
            look_ahead_mask=la_mask,
            enc_dec_mask=s_mask
        )
        # Loss 계산
        if loss_function is not None:
            loss = loss_function(pred, y_input)
            loss_sum += loss.item()
        
        # 모드 선택
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        elif mode in ['val', 'test']:
            pass
        else:
            raise ValueError('Mode should be either train, val, or test')

        # 결과값 저장
        logit = pred.to('cpu').detach().numpy() # 메모리 해제
        logit_list.extend(logit)

        # label 값 저장
        b_label_list.append(y_input)

        if purpose == 'classification':
            # 1순위 및 신뢰도
            max_index = np.argmax(logit, axis=1)
            pred_label_list.append(max_index)
            
            # 2순위 및 신뢰도
            max_2nd_index, pred_reliability = cfum.get_max_2nd_n_reliability(logit)
            pred_2nd_label_list.append(max_2nd_index)
            pred_reliability_list.append(pred_reliability)
            
        iter_n += 1
        
    # 결과 변수
    try:
        logit_ary = np.array(logit_list)
        b_label_ary = torch.concat(b_label_list).to('cpu').detach().numpy()
    except KeyError:
        print('에러가 발생하였습니다.')
        sys.exit()
    
    if purpose == 'classification':
        pred_label_ary = np.concatenate(pred_label_list)
        pred_reliability_ary = np.concatenate(pred_reliability_list)
        pred_2nd_label_ary = np.concatenate(pred_2nd_label_list)
        result_list.append(pred_label_ary)
        result_list.append(pred_reliability_ary)
        result_list.append(pred_2nd_label_ary)
    
    running_loss = loss_sum / iter_n
    
    torch.cuda.empty_cache()
    return logit_ary, b_label_ary, running_loss, result_list



# class Iterator():
#     def __init__(self, dataloader, model, device):
#         self.model = model
#         self.device = device

#         self.batch_idx_list = []
#         self.x_list = []
#         self.x_mark_list = []
#         self.y_list = []
#         self.y_mark_list = []
#         self.s_mask_list = []
#         self.la_mask_list = []
        
#         for batch_idx, (x, x_mark, s_mask, y, y_mark, la_mask) in enumerate(tqdm(dataloader)):
#             self.batch_idx_list.append(batch_idx)
#             self.x_list.append(x)
#             self.x_mark_list.append(x_mark)
#             self.y_list.append(y)
#             self.y_mark_list.append(y_mark)
#             self.s_mask_list.append(s_mask)
#             self.la_mask_list.append(la_mask)
#         # self.batch_idx_list = batch_idx_list
#         # self.x_list = x_list
#         # self.x_mark_list = x_mark_list
#         # self.s_mask_list = s_mask_list
        
#         self.count = -1
            
#     def __len__(self):
#         return len(self.batch_idx_list)
        
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.count < len(self.batch_idx_list) - 1:
#             self.count += 1
#             x = self.x_list[self.count]
#             x_input = x.to(self.device, dtype=torch.float)
            
#             x_mark = self.x_mark_list[self.count]
#             x_mark = torch.tensor(x_mark, dtype=torch.float32)
#             x_mark = x_mark.to(self.device)
            
#             s_mask = self.s_mask_list[self.count]
#             s_mask = torch.tensor(s_mask)
#             s_mask = s_mask.to(self.device)
            
#             y = self.y_list[self.count]
#             y_input = y.to(self.device, dtype=torch.float)
            
#             y_mark = self.y_mark_list[self.count]
#             y_mark = torch.tensor(y_mark, dtype=torch.float32)
#             y_mark = y_mark.to(self.device)
            
#             la_mask = self.la_mask_list[self.count]
#             la_mask = torch.tensor(la_mask)
#             la_mask = la_mask.to(self.device)
            
#             pred = self.model(
#                 x=x_input,
#                 x_mark=x_mark, 
#                 y=y_input, 
#                 y_mark=y_mark,
#                 enc_self_mask=s_mask,
#                 look_ahead_mask=la_mask,
#                 enc_dec_mask=s_mask
#                 )
            
#             logit = pred
#             b_label = y_input
#             return pred, logit, b_label
#         else:
#             raise StopIteration



    





# # # 전체입력
# # class Iterator():
# #     def __init__(self, dataloader, model, device):
# #         self.dataloader = dataloader
# #         self.model = model
# #         self.device = device
# #         self.count = -1
# #         self.batch_idx_list = []
# #         self.x_list = []
# #         self.x_mask_list = []
# #         self.b_label_list = []
# #         for batch_idx, (x, x_mask, b_label) in enumerate(dataloader):
# #             self.batch_idx_list.append(batch_idx)
# #             self.x_list.append(x)
# #             self.b_label_list.append(b_label)
# #             self.x_mask_list.append(x_mask)
        
            
# #     def __len__(self):
# #         return len(self.dataloader)


# #     def __iter__(self):
# #         return self

    
# #     def __next__(self):
# #         if self.count < len(self.batch_idx_list) - 1:
# #             self.count += 1
            
# #             # x
# #             x = self.x_list[self.count]
# #             x = x.to(self.device, dtype=torch.float32)
            
# #             # x mask
# #             x_mask = self.x_mask_list[self.count]
# #             while len(x_mask.size()) < 4:
# #                 x_mask = x_mask.unsqueeze(1)
# #             x_mask = x_mask.to(self.device)
            
# #             # b label
# #             b_label = self.b_label_list[self.count]
# #             b_label = b_label.to(self.device, dtype=torch.float32)
# #             # # b label mask
# #             # b_label_mask = self.b_label_mask_list[self.count]
# #             # while len(b_label_mask.size()) < 4:
# #             #     b_label_mask = b_label_mask.unsqueeze(1)
# #             # b_label_mask = b_label_mask.to(self.device)
            
# #             # output
# #             pred = self.model(
# #                 input=x,
# #                 input_mask=x_mask,
# #             )
# #             del x, x_mask
# #             return pred, b_label
# #         else:
# #             raise StopIteration


