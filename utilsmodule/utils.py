import numpy as np
import pandas as pd
import os
import sys
import json
import time

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


def get_logger(get, log_file_name, time_handler=False, console_display=False, logging_level='info'):
    '''
    로거 함수

    parameters
    ----------
    log_file_name: str
        logger 파일을 생성할 때 적용할 파일 이름.
    
    logging_level: str
        logger 를 표시할 수준. (notset < debug < info < warning < error < critical)
    
    returns
    -------
    logger: logger
        로거를 적용할 수 있는 로거 변수
    '''
    import logging
    from logging import handlers

    logger = logging.getLogger(get)
    if logging_level == 'critical':
        logger.setLevel(logging.CRITICAL)
    if logging_level == 'error':
        logger.setLevel(logging.ERROR)
    if logging_level == 'warning':
        logger.setLevel(logging.WARNING)
    if logging_level == 'info':
        logger.setLevel(logging.INFO)
    if logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
    if logging_level == 'notset':
        logger.setLevel(logging.NOTSET)
    
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s level:%(levelname)s %(filename)s line %(lineno)d %(message)s')
    if console_display:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if time_handler:
        file_handler = handlers.TimedRotatingFileHandler(
            filename=log_file_name,        
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8")
        file_handler.suffix = '%Y%m%d'
    else:
        file_handler = logging.FileHandler(log_file_name)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# def make_class_dict(class_list):
#     '''
#     클래스 리스트를 기준으로 인코딩을 적용하여 {'클래스' : idx} 형태의 클래스 dictionary 를 만드는 함수.

#     parameters
#     ----------
#     class_list: list, shape=(n, )
#         클래스 값 들로 구성된 리스트.

#     returns
#     -------
#     class_dict: dictionary
#         {'클래스' : idx} 형태의 dictionary.

#     '''
#     class_dict = {}
#     for idx, class_name in enumerate(class_list):
#         class_dict[class_name] = idx
#     return class_dict


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
    
    
def get_save_path(args):
    args_setting = vars(args)
    os.makedirs(f'{args.root_path}/{args.phase}/{args.dataset_name}_model', exist_ok=True)
    condition_list = os.listdir(f'{args.root_path}/{args.phase}/{args.dataset_name.split(".")[0]}_model')

    if len(condition_list) == 0:
        condition_order = len(condition_list)
    else:
        args_key_list = list(args_setting.keys())

        for condition in condition_list:
            with open(f'{args.root_path}/{args.phase}/{args.dataset_name.split(".")[0]}_model/{condition}/args_setting_{condition}.json', 'r', encoding='utf-8-sig') as f:
                exist_args_setting = json.load(f)

            exist_args_key_list = list(exist_args_setting.keys())

            for key, e_key in zip(args_key_list, exist_args_key_list):
                if (key == 'epochs'):
                    continue
                if key == e_key:
                    if args_setting[key] != exist_args_setting[e_key]:
                        condition_order = int(condition) + 1
                        break
                    else:
                        condition_order = int(condition)
                        continue
                if key != e_key:
                    condition_order = file_order = int(condition) + 1
                    break
            if condition_order > int(condition):
                continue
            else:
                break

    condition_order = str(condition_order).zfill(4)
    model_save_path = f'{args.root_path}/{args.phase}/{args.dataset_name.split(".")[0]}_model/{condition_order}'

    # args setting 저장
    os.makedirs(model_save_path, exist_ok=True)
    with open(f'{model_save_path}/args_setting_{condition_order}.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    print(f'train condition : {condition_order}')
    return model_save_path


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
    h = t // 3600
    m = (t % 3600) // 60
    s = int(t % 60)
    return h, m ,s
    