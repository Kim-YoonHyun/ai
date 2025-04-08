'''
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install transformers
'''
import sys
import os
sys.path.append(os.getcwd())
import json
import copy
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch

from mylocalmodules import dataloader as dam
from models import network as net

sys.path.append('/home/kimyh/python/ai')
from sharemodule import lossfunction as lfm
from sharemodule import logutils as lom
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm


def get_args():
    parser = argparse.ArgumentParser()
    
    # basic parameter
    parser.add_argument('--root_path')
    parser.add_argument('--root_dataset_path')
    parser.add_argument('--phase')
    parser.add_argument('--dataset_name')
    parser.add_argument('--purpose', type=str)
        
    # train parameter
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function_name', type=str, default='MSE')
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--scheduler_name', type=str, default='CosineAnnealingLR')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sampler_name', type=str, default='SequentialSampler')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--pre_trained', type=str, default='None')
    args = parser.parse_args()
    return args


def trainer(args_setting):
    
    # =========================================================================
    # 데이터 셋 설정
    dataset_path = f"{args_setting['root_dataset_path']}/datasets/{args_setting['dataset_name']}"
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dam.get_dataloader(
        input1=input1, 
        input2=inptu2, 
        batch_size=args_setting['batch_size'], 
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'],
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    print('validation dataloader 생성 중...')
    Val_Dataloader = dam.get_dataloader(
        input1=input1, 
        input2=inptu2, 
        batch_size=args_setting['batch_size'], 
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'],
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    # =========================================================================
    # device 
    tum.see_device()
    device = tum.get_device(args_setting['device_num'])
    
    # =========================================================================
    # model        
    model = net.name()
    model.to(device)
    
    # pre-trained
    torch.autograd.set_detect_anomaly(True)
    if args_setting['pre_trained'] != 'None':
        model.load_state_dict(torch.load(f"{args_setting['pre_trained']}"))
        print('\n>>> use pre-trained model')
    else:
        start_epoch = 0
        
    # see parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"총 파라미터 수: {total_params}")
    # sys.exit()

    # =========================================================================
    # optimizer
    optimizer = tum.get_optimizer(
        base='torch',
        method=args_setting['optimizer_name'],
        model=model,
        learning_rate=args_setting['learning_rate']
    )

    # loss function
    loss_function = lfm.LossFunction(
        base='torch',
        method=args_setting['loss_function_name']
    )
    # scheduler
    scheduler = tum.get_scheduler(
        base='torch',
        method=args_setting['scheduler_name'],
        optimizer=optimizer
    )
    # =========================================================================
    # save setting
    root_save_path = f"{args_setting['root_path']}/trained_model/{args_setting['phase']}/{args_setting['dataset_name']}"
    condition_order = tum.get_condition_order(
        args_setting=args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs', 'device_num', 'batch_size']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose=args_setting['purpose'], 
        start_epoch=0, 
        epochs=args_setting['epochs'], 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=args_setting['max_grad_norm'], 
        reset_class=None, 
        model_save_path=model_save_path
    )
    print(condition_order)


def main():
    args = get_args()
    args_setting = vars(args)

    # =========================================================================
    utm.envs_setting(args_setting['random_seed'])

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=args_setting['root_path'],
        log_file_name='train.log',
        time_handler=True
    )
    # =========================================================================
    # 학습 진행
    trainer(args_setting)

if __name__ == "__main__":
    main()    