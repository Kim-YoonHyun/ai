import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')






import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')


sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tru
# import psutil
# def print_memory_usage():
#     memory_info = psutil.virtual_memory()
#     print(f'CPU Memory Usage: {memory_info.percent}% Used, {memory_info.available / (1024 ** 3):.2f} GB available')


def iterator(mode, purpose, dataloader, model, device, criterion=None, optimizer=None, max_grad_norm=None):
    
    # loss, 결과 초기화
    logit_list = []
    b_label_list = []
    loss_sum = 0
    result_list = []

    iter_n = 0
    
    # 이터레이터 진행
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        
        # 데이터 tensor 에 올리기
        x_input = torch.tensor(x).to(device)#, dtype=torch.float)
        y_input = torch.tensor(y).to(device)#, dtype=torch.float)
        
        # 모델에 입력
        pred = model(
            x=x_input,
            y=y_input
        )
        
        # Loss 계산
        if criterion is not None:
            loss = criterion(pred, y_input)
            
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
            pass
            # 결과값 저장
            # logit = pred.to('cpu').detach().numpy() # GPU --> CPU 
            # logit_list.extend(logit)

            # label 값 저장
            # b_label_list.append(y_input)
        else:
            raise ValueError('Mode should be either train, val, or test')

        iter_n += 1
        # tru.print_memory_usage()
        # tru.get_gpu_temperature()
        tru.gpu_temp_control()
        
    # 결과 변수
    try:
        logit_ary = np.array(logit_list)
        b_label_ary = torch.concat(b_label_list).to('cpu').detach().numpy()
    except Exception:
        logit_ary = np.array([])
        b_label_ary = np.array([])
    
    # 최종 학습 loss 계산
    running_loss = loss_sum / iter_n
    
    # GPU 메모리 캐시 제거    
    torch.cuda.empty_cache()
    return logit_ary, b_label_ary, running_loss, result_list








# def iterator(mode, purpose, dataloader, model, device, criterion=None, optimizer=None, max_grad_norm=None):
    
#     # loss, 결과 초기화
#     logit_list = []
#     b_label_list = []
#     loss_sum = 0
#     result_list = []

#     iter_n = 0
    
#     # 이터레이터 진행
#     for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
#         x_input = x.to(device, dtype=torch.float)
#         y_input = y.to(device, dtype=torch.float)
        
#         pred = model(
#             x=x_input,
#             y=y_input
#         )
        
#         # Loss 계산
#         if criterion is not None:
#             loss = criterion(pred, y_input)
#             loss_sum += loss.item()
        
#         # 모드 선택
#         if mode == 'train':
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             optimizer.step()
#         elif mode in ['val', 'test']:
#             pass
#         else:
#             raise ValueError('Mode should be either train, val, or test')

#         # 결과값 저장
#         logit = pred.to('cpu').detach().numpy() # 메모리 해제
#         logit_list.extend(logit)

#         # label 값 저장
#         b_label_list.append(y_input)

#         iter_n += 1
        
#     # 결과 변수
#     try:
#         logit_ary = np.array(logit_list)
#         b_label_ary = torch.concat(b_label_list).to('cpu').detach().numpy()
#     except KeyError:
#         logit_ary = np.array([])
#         b_label_ary = np.array([])
    
#     running_loss = loss_sum / iter_n
    
#     torch.cuda.empty_cache()
#     return logit_ary, b_label_ary, running_loss, result_list



