'''
pip install xlrd
'''
import numpy as np
import pandas as pd
import shutil
import os
import sys
import json
import time
import csv
from tqdm import tqdm
from datetime import date, datetime, timedelta


def save_yaml(path, obj):
    import yaml
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)


def load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def envs_setting(random_seed):
    '''
    난수지정 등의 환경설정

    parameters
    ----------
    random_seed: int
        설정할 random seed

    returns
    -------
    torch, numpy, random 등에 대한 랜덤 시드 고정    
    '''

    import torch
    import torch.backends.cudnn as cudnn
    import random
    import numpy as np
    
    
    # seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(random_seed)
    random.seed(random_seed)


def normalize_1D(ary):
    '''
    1차원데이터를 0~1 사이 값으로 normalize 하는 함수

    parameters
    ----------
    ary: numpy array
        noramlize 를 적용할 1차원 array
    
    returns
    -------
    0 ~ 1 사이로 noramalize 된 array
    '''
    ary = np.array(ary)
    
    if len(ary.shape) > 1:
        return print('1 차원 데이터만 입력 가능')
    
    ary_min = np.min(ary)
    ary_min = np.subtract(ary, ary_min)
    ary_max = np.max(ary_min)
    ary_norm = np.divide(ary_min, ary_max)
    
    return ary_norm
    

def get_error_info():
    import traceback
    traceback_string = traceback.format_exc()
    return traceback_string


def read_jsonl(data_path):
    try:
        data_list = validate_data(
            data_path=data_path,
            encoding='utf-8-sig'
        )
        
    except UnicodeDecodeError:
        
        data_list = validate_data(
            data_path=data_path,
            encoding='cp949'
        )
    return data_list
    
    
def validate_data(data_path, encoding):
    data_list = []
    try:
        with open(data_path, 'r', encoding=encoding) as f:
            prodigy_data_list = json.load(f)
        data_list.append(prodigy_data_list)
    except json.decoder.JSONDecodeError:
        with open(data_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.replace('\n', '')
                line.strip()
                if line[-1] == '}':
                    json_line = json.loads(line)
                    data_list.append(json_line)
    return data_list


def tensor2array(x_tensor):
    x_ary = x_tensor.detach().cpu().numpy()
    return x_ary


def save_tensor(x_tensor, mode):
    x_ary = tensor2array(x_tensor=x_tensor)
    
    if mode == 1:
        b = x_ary[0]
        # b = np.round(b, 3)
        b = np.where(np.absolute(b) > 2, np.round(b, 0), np.round(b, 3))
        df = pd.DataFrame(b)
        df.to_csv(f'./temp.csv', index=False, encoding='utf-8-sig')
        print(df)
        print(x_ary.shape)
    
    if mode == 2:
        ary = x_ary[0]
        i, j, k = ary.shape
        print(i, j, k)
        for idx in range(k):
            a = np.squeeze(ary[:, :, idx:idx+1])
            a = np.where(np.absolute(a) > 2, np.round(a, 0), np.round(a, 3))
            df = pd.DataFrame(a)
            df.to_csv(f'./temp{idx}.csv', index=False, encoding='utf-8-sig')
            print(df)
        print(x_ary.shape)


# def identify_nan_section(ary, stan_num, mode, reverse=False):
#     nan_start_idx = 0
#     nan_start_idx_list = []
#     nan_end_idx_list = []
#     flag = 1
#     if len(ary) == 0:
#         return [], []
#     for idx, value in enumerate(ary):
        
#         value_str = str(value)
        
#         # 가장 처음인 경우
#         if idx == 0:
#             pre_value = value_str
#             continue
        
#         # 현재 값이 nan 일 때
#         if value_str == 'nan':
#             # 이전값이 nan 인경우
#             if pre_value == 'nan':
#                 flag += 1
#             # 이전 값이 nan 이 아닌 경우
#             else:
#                 flag = 1
#                 # idx 시작 위치 지정
#                 nan_start_idx = idx
                
#         # 현재 값이 nan 이 아닐 때
#         else:
#             # 이전 값이 nan 인 경우
#             if pre_value == 'nan':
#                 # idx 끝 위치 지정
#                 if mode == 'above':
#                     if flag >= stan_num:
#                         nan_start_idx_list.append(nan_start_idx)
#                         nan_end_idx_list.append(idx-1)
#                 elif mode == 'below':
#                     if flag <= stan_num:
#                         nan_start_idx_list.append(nan_start_idx)
#                         nan_end_idx_list.append(idx-1)
#                 else:
#                     print('mode 를 above 또는 이하 below 중 하나로 지정해주세요')
#                     raise KeyError()
#         pre_value = value_str
        
#     if value_str == 'nan':
#         if mode == 'above':
#             if flag >= stan_num:
#                 nan_start_idx_list.append(nan_start_idx)
#                 nan_end_idx_list.append(idx)
#         elif mode == 'below':
#             if flag <= stan_num:
#                 nan_start_idx_list.append(nan_start_idx)
#                 nan_end_idx_list.append(idx)
#         else:
#             print('mode 를 above 또는 이하 below 중 하나로 지정해주세요')
#             raise KeyError()
    
#     if reverse:
#         rev_start_idx_list = [0]
#         rev_end_idx_list = [len(ary)-1]
#         for ns_idx, ne_idx in zip(nan_start_idx_list, nan_end_idx_list):
#             if ns_idx == 0:
#                 rev_start_idx_list.pop(0)
#                 rev_start_idx_list.append(ne_idx+1)
#                 continue
#             if ne_idx == len(ary)-1:
#                 rev_end_idx_list.pop(-1)
#                 rev_end_idx_list.append(ns_idx-1)
#                 continue
#             rev_start_idx_list.append(ne_idx+1)
#             rev_end_idx_list.insert(-1, ns_idx-1)
#         return rev_start_idx_list, rev_end_idx_list

#     return nan_start_idx_list, nan_end_idx_list



def identify_repeat_section(ary, stan_num, include_nan=False):
    start_idx_list = []
    end_idx_list = []
    start_idx = 0
    # pre_value = 'nan'
    flag = 1    
    for idx, value in enumerate(ary):
        value_str = str(value)
        
        # 가장 처음인 경우
        if idx == 0:
            pre_value = value_str
            continue
        
        # 현재 값이 이전 값과 동일할때
        if value_str == pre_value:
            # 현재 값이 nan 이 아닌 경우 만
            
            if include_nan:
                flag += 1
            else:
                if value_str != 'nan':
                    flag += 1
        # 현재 값이 이전 값과 다를때
        else:
            if flag >= stan_num:
                start_idx_list.append(start_idx)
                end_idx_list.append(idx-1)
                
            # 시작 지점 갱신
            start_idx = idx
            flag = 1
        pre_value = value_str
    
    if flag >= stan_num:
        start_idx_list.append(start_idx)
        end_idx_list.append(idx)
    return start_idx_list, end_idx_list


def identify_stan_repeat_section(ary, stan_value, stan_repeat, mode, reverse=False):
    nan_start_idx = 0
    nan_start_idx_list = []
    nan_end_idx_list = []
    flag = 1
    if len(ary) == 0:
        return [], []
    for idx, value in enumerate(ary):
        
        value_str = str(value)
        
        # 가장 처음인 경우
        if idx == 0:
            pre_value = value_str
            continue
        
        # 현재 값이 stan 일 때
        if value_str == stan_value:
            # 이전값이 nan 인경우
            if pre_value == stan_value:
                flag += 1
            # 이전 값이 nan 이 아닌 경우
            else:
                flag = 1
                # idx 시작 위치 지정
                nan_start_idx = idx
                
        # 현재 값이 nan 이 아닐 때
        else:
            # 이전 값이 nan 인 경우
            if pre_value == stan_value:
                # idx 끝 위치 지정
                if mode == 'above':
                    if flag >= stan_repeat:
                        nan_start_idx_list.append(nan_start_idx)
                        nan_end_idx_list.append(idx-1)
                elif mode == 'below':
                    if flag <= stan_repeat:
                        nan_start_idx_list.append(nan_start_idx)
                        nan_end_idx_list.append(idx-1)
                else:
                    print('mode 를 above 또는 이하 below 중 하나로 지정해주세요')
                    raise KeyError()
        pre_value = value_str
        
    if value_str == stan_value:
        if mode == 'above':
            if flag >= stan_repeat:
                nan_start_idx_list.append(nan_start_idx)
                nan_end_idx_list.append(idx)
        elif mode == 'below':
            if flag <= stan_repeat:
                nan_start_idx_list.append(nan_start_idx)
                nan_end_idx_list.append(idx)
        else:
            print('mode 를 above 또는 이하 below 중 하나로 지정해주세요')
            raise KeyError()
    
    if reverse:
        rev_start_idx_list = [0]
        rev_end_idx_list = [len(ary)-1]
        for ns_idx, ne_idx in zip(nan_start_idx_list, nan_end_idx_list):
            if ns_idx == 0:
                rev_start_idx_list.pop(0)
                rev_start_idx_list.append(ne_idx+1)
                continue
            if ne_idx == len(ary)-1:
                rev_end_idx_list.pop(-1)
                rev_end_idx_list.append(ns_idx-1)
                continue
            rev_start_idx_list.append(ne_idx+1)
            rev_end_idx_list.insert(-1, ns_idx-1)
        return rev_start_idx_list, rev_end_idx_list

    return nan_start_idx_list, nan_end_idx_list