import sys
import os
import argparse
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 
import torch

# local modules
from models import network as net
from mylocalmodules import dataloader as dam

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import utils as utm


def main():
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')

    args = parser.parse_args()
    args_setting = vars(args)
    utm.envs_setting(42)
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
    device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    shuffle = args_setting['shuffle']
    drop_last = args_setting['drop_last']
    num_workers = args_setting['num_workers']
    pin_memory = args_setting['pin_memory']
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    # =========================================================================
    # dataset 불러오기
    input_data_list = []
    
    # =========================================================================
    print('get dataloader')
    Test_Dataloader = dam.get_dataloader(
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
    weight = torch.load(f'{trained_model_path}/{trained_model}/weight.pt')
    model.load_state_dict(weight)
    model.to(device)
    
    # ===================================================================    
    model.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(Test_Dataloader):
            x = x.to(device, dtype=torch.long)

            # pred
            pred = model(x)
            pred = pred.to('cpu').detach().numpy() # 메모리 해제
            
            # if classification
            max_index_list = np.argmax(pred, axis=1).tolist()

        
if __name__  == '__main__':
    main()