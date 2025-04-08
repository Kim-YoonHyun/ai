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
    for batch_idx, x, y in enumerate(tqdm(dataloader)):
        x_input = x.to(device, dtype=torch.float)
        y_input = y.to(device, dtype=torch.float)

        pred = model(x=x_input)
        
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
