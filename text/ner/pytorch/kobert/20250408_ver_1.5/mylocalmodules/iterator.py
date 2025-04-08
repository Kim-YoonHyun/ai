import sys
import os
import numpy as np
from tqdm import tqdm
import time
import torch
import warnings
warnings.filterwarnings('ignore')


sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tru


# 과열현상 해결방법 진행중
#==
'''
nvidia-smi -pl 200  # 200W로 제한 (기본값보다 낮게 설정)
'''

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
#==


def iterator(mode, purpose, dataloader, model, device, criterion=None, optimizer=None, max_grad_norm=None):
    
    # loss, 결과 초기화
    logit_list = []
    b_label_list = []
    loss_sum = 0
    result_list = []

    iter_n = 0
    
    # 이터레이터 진행
    # for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
    for batch_idx, (input_ids, attention_mask, token_type_ids, label_ids) in enumerate(tqdm(dataloader)):
        
        # input ids
        input_ids = input_ids.to(device)
        
        # attention_mask
        attention_mask = attention_mask.to(device)
        
        # token type ids
        token_type_ids = token_type_ids.to(device)
        
        # labels
        labels = label_ids.to(device)
        
        # input 생성
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        inputs['token_type_ids'] = token_type_ids
        
        pred = model(**inputs)
        _, logit = pred[:2]
        b_label = labels
        
        # Loss 계산
        if criterion is not None:
            logit = logit.permute(0, 2, 1)
            loss = criterion(logit, b_label)
            
            # loss 합산            
            loss_sum += loss.item()
        
        # 모드 선택
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            '''
            Gradient Explosion (기울기 폭발) 문제를 방지하기 위한 함수
            기울기 벡터의 크기가 max_grad_norm 을 넘길 경우 모든 기울기를 max_norm 에 맟춰 축소하여 일정 크기 이하로 제한
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
        elif mode in ['val', 'test']:
            # 결과값 저장
            logit = logit.to('cpu').detach().numpy() # GPU --> CPU 
            logit_list.extend(logit)

            # label 값 저장
            b_label = b_label.to('cpu').detach().numpy()
            b_label_list.append(b_label)
        else:
            raise ValueError('Mode should be either train, val, or test')

        iter_n += 1
        # tru.print_memory_usage()
        # tru.get_gpu_temperature()
        tru.gpu_temp_control()
        
        
    # 결과 변수
    try:
        logit_ary = np.array(logit_list)
        b_label_ary = np.array(b_label_list)
    except Exception:
        logit_ary = np.array([])
        b_label_ary = np.array([])
    
    # 최종 학습 loss 계산
    running_loss = loss_sum / iter_n
    
    # GPU 메모리 캐시 제거    
    torch.cuda.empty_cache()
    return logit_ary, b_label_ary, running_loss, result_list