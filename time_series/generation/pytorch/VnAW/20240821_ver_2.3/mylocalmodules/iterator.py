import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')


def iterator(mode, purpose, dataloader, model, device, loss_function=None, optimizer=None, max_grad_norm=None):
    
    # loss, 결과 초기화
    logit_list = []
    b_label_list = []
    loss_sum = 0
    result_list = []

    iter_n = 0
    
    # 이터레이터 진행
    for batch_idx, (x, x_mark, x_s_mask, y, y_mark, e_d_mask, y_la_mask) in enumerate(tqdm(dataloader)):
        x_input = x.to(device, dtype=torch.float)
        y_input = y.to(device, dtype=torch.float)
        
        x_mark = torch.tensor(x_mark, dtype=torch.float32)
        y_mark = torch.tensor(y_mark, dtype=torch.float32)
        x_mark = x_mark.to(device)
        y_mark = y_mark.to(device)
        
        x_s_mask = torch.tensor(x_s_mask)
        x_s_mask = x_s_mask.to(device)
        
        e_d_mask = torch.tensor(e_d_mask)
        e_d_mask = e_d_mask.to(device)
        
        y_la_mask = torch.tensor(y_la_mask)
        y_la_mask = y_la_mask.to(device)
        pred = model(
            x=x_input,
            y=y_input, 
            x_mark=x_mark, 
            y_mark=y_mark,
            enc_self_mask=x_s_mask,
            look_ahead_mask=y_la_mask,
            enc_dec_mask=e_d_mask
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

        iter_n += 1
        
    # 결과 변수
    try:
        logit_ary = np.array(logit_list)
        b_label_ary = torch.concat(b_label_list).to('cpu').detach().numpy()
    except KeyError:
        print('에러가 발생하였습니다.')
        sys.exit()
    
    running_loss = loss_sum / iter_n
    
    torch.cuda.empty_cache()
    return logit_ary, b_label_ary, running_loss, result_list



