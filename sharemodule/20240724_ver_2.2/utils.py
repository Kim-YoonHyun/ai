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
    
    
def see_device():
    '''
    선택 가능한 gpu device 표시
    '''
    import torch
    if torch.cuda.is_available():
        print('\n------------- GPU list -------------')
        n_devices = torch.cuda.device_count()
        for i in range(n_devices):
            print(f'{i}: ', torch.cuda.get_device_name(i))
        print('------------------------------------')
    else:
        print('No GPU available')   


def get_error_info():
    import traceback
    traceback_string = traceback.format_exc()
    return traceback_string


def time_measure(start):
    t = time.time() - start
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return h, m ,s
    

def read_df(path):
    extention = path.split('.')[-1]
    if extention in ['csv', 'CSV']:
        switch = 'csv'
    elif extention in ['xlsx', 'xls']:
        switch = 'excel'
    elif extention in ['txt']:
        switch = 'txt'
    else:
        raise ValueError(f'{extention}은(는) 잘못되거나 지정되지 않은 확장자입니다.')
    
    if switch == 'csv':
        encoding = 'utf-8-sig'
        while True:
            try:
                data_df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                encoding = 'cp949'
            except pd.errors.ParserError:
                f = open(path, encoding=encoding)
                reader = csv.reader(f)
                csv_list = []
                for line in reader:
                    if len(line) != 38:
                        pass
                    csv_list.append(line)
                f.close()
                data_df = pd.DataFrame(csv_list)
                data_df.columns = data_df.iloc[0].to_list()
                data_df = data_df.drop(index=data_df.index[0])	# 0번째 행을 지움
                break
    if switch == 'excel':
        data_df = pd.read_excel(path)
    if switch == 'txt':
        line_list = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_list.append(line)
        data_df = pd.DataFrame(line_list, columns=['string'])
    return data_df


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


def get_date_list(schedule, year, mon_list, start_day_list, end_day_list):
    date_list = []
    if schedule:
        yesterday = date.today() - timedelta(1)
        yesterday = str(yesterday)
        date_list = [yesterday]
    else:
        for mon in mon_list:
            start_day = start_day_list[mon-1]
            end_day = end_day_list[mon-1]
            for dd in range(start_day, end_day+1):
                dd = str(dd).zfill(2)
                mm = str(mon).zfill(2)
                date_list.append(f'{year}-{mm}-{dd}')
    return date_list