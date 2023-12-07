import numpy as np
import pandas as pd
import shutil
import os
import sys
import json
import time
import csv
from tqdm import tqdm

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
    

def get_condition_order(args_setting, save_path, except_arg_list=None):
    os.makedirs(save_path, exist_ok=True)
    condition_list = os.listdir(save_path)
    condition_list.sort()
    if len(condition_list) == 0:
        condition_order = 0
    else:
        args_key_list = list(args_setting.keys())
        args_key_list.sort()

        for condition in condition_list:
            with open(f'{save_path}/{condition}/args_setting.json', 'r', encoding='utf-8-sig') as f:
                exist_args_setting = json.load(f)

            exist_args_key_list = list(exist_args_setting.keys())
            exist_args_key_list.sort()

            for key, e_key in zip(args_key_list, exist_args_key_list):
                if except_arg_list is not None:
                    if key in except_arg_list:
                        continue
                if key == e_key:
                    if args_setting[key] != exist_args_setting[e_key]:
                        condition_order = int(condition) + 1
                        break
                    else:
                        condition_order = int(condition)
                        continue
                if key != e_key:
                    condition_order = int(condition) + 1
                    break
            if condition_order > int(condition):
                continue
            else:
                break

    condition_order = str(condition_order).zfill(4)

    return condition_order


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