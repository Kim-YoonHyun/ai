import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import argparse

import torch

from models import network as net

# local modules
from mylocalmodules import dataloader as dam

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
# from sharemodule import classificationutils as clm
from sharemodule import utils as utm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')
    parser.add_argument('--device_num')
    parser.add_argument('--dummy_label', type=bool)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--result_save_path', type=str)

    args = parser.parse_args()
    args_setting = vars(args)
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    device_num = args.device_num
    dummy_label = args.dummy_label
    test_data = args.test_data
    result_save_path = args.result_save_path
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TEST',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    
    # ==========================================================
    # 모델에 적용된 변수 불러오기
    print('변수 불러오기')
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
    
    try:
        device_num = int(device_num)
    except ValueError:
        device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    shuffle = args_setting['shuffle']
    drop_last = args_setting['drop_last']
    num_workers = args_setting['num_workers']
    pin_memory = args_setting['pin_memory']
    random_seed = args_setting['random_seed']
    
    
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # dataset 불러오기
    input1_data_list = []
    input2_data_list = []
    
    # =========================================================================
    print('get dataloader')
    Test_Dataloader = dam.get_dataloader(
        input1=input1_data_list,
        input2=input2_data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    # ===================================================================    
    print('get device')
    device = tum.get_device(device_num)
    
    print('get_model')
    model = net()
    
    # 학습된 가중치 로딩
    weight_path = f'{trained_model_path}/{trained_model}/weight.pt'
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    logit_ary, true_label_ids_ary, _, _ = trm.model_test(
        model=model, 
        test_dataloader=Test_Dataloader, 
        device=device
    )
    
    

        
if __name__  == '__main__':
    main()
