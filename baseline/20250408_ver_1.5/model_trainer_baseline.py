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

from mylocalmodules import dataloader as dl
from model import network as net

sys.path.append('/home/kimyh/python/ai')
from sharemodule import lossfunction as lf
from sharemodule import logutils as lu
from sharemodule import train as tr
from sharemodule import trainutils as tru
from sharemodule import utils as u


def get_args():
    parser = argparse.ArgumentParser()
    
    # basic parameter
    parser.add_argument('--root_path')
    parser.add_argument('--root_dataset_path')
    parser.add_argument('--phase')
    parser.add_argument('--dataset_name')
    parser.add_argument('--reduce_num', type=int, default=0)
    parser.add_argument('--scale', type=str, default='fixed-max')
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
    
    # dataset parameter
    parser.add_argument('--train_p', type=float, default=0.1)
    parser.add_argument('--sampler_name', type=str, default='SequentialSampler')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--pre_trained', type=str, default='None')
    
    args = parser.parse_args()
    return args


def reducing(data_list, total_num, reduce_num=0):
    if reduce_num == 0:
        size = total_num
    else:
        size = reduce_num
    filter_list = np.array(data_list)[:size].tolist()
    return filter_list

    
def trainer(ars):
    # =========================================================================
    # 데이터 셋 경로
    dataset_path = f"{ars['root_dataset_path']}/datasets/{ars['dataset_name']}"
    
    '''
    dataset code
    '''
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dl.get_dataloader(
        mode='train',
        batch_size=ars['batch_size'],
        dataset_path=f'{dataset_path}/train',
        x_name_list=None,
        y_name_list=None,
        scale=ars['scale'],
        norm=None,
        shuffle=ars['shuffle'], 
        drop_last=ars['drop_last'],
        num_workers=ars['num_workers'], 
        pin_memory=ars['pin_memory'], 
        sampler_name=ars['sampler_name']
    )
    print('validation dataloader 생성 중...')
    Val_Dataloader = dl.get_dataloader(
        mode='train',
        batch_size=ars['batch_size'], 
        dataset_path=f'{dataset_path}/val', 
        x_name_list=None, 
        y_name_list=None, 
        scale=ars['scale'],
        norm=None,
        shuffle=ars['shuffle'], 
        drop_last=ars['drop_last'], 
        num_workers=ars['num_workers'], 
        pin_memory=ars['pin_memory'], 
        sampler_name=ars['sampler_name']
    )
    
    # =========================================================================
    # device 
    tru.see_device()
    device = tru.get_device(ars['device_num'])
    
    # =========================================================================
    # model        
    model = net.network()
    model.to(device)

    # pre-trained
    start_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    if ars['pre_trained'] != 'None':
        model.load_state_dict(torch.load(f"{ars['pre_trained']}"))
        print('\n>>> use pre-trained model')
        start_epoch = 0
        
    # 파라미터 갯수 출력
    tru.see_parameters(model)
    
    # =========================================================================
    # optimizer
    optimizer = tru.get_optimizer(
        base='torch',
        method=ars['optimizer_name'],
        model=model,
        learning_rate=ars['learning_rate']
    )

    # loss function
    criterion = lf.LossFunction(
        base='torch',
        method=ars['loss_function_name']
    )
    # scheduler
    scheduler = tru.get_scheduler(
        base='torch',
        method=ars['scheduler_name'],
        optimizer=optimizer
    )
    
    # =========================================================================
    # save setting
    root_save_path = f"{ars['root_path']}/trained_model/{ars['phase']}/{ars['dataset_name']}"
    condition_order = tru.get_condition_order(
        args_setting=ars,
        save_path=root_save_path,
        except_arg_list=['epochs', 'batch_size']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(ars, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = tr.train(
        model=model, 
        purpose=ars['purpose'], 
        start_epoch=start_epoch, 
        epochs=ars['epochs'], 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        device=device, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=ars['max_grad_norm'], 
        model_save_path=model_save_path,
        argmax_axis=1,
        label2id_dict=None, 
        id2label_dict=None
    )
    
    print(condition_order)


def main():
    args = get_args()
    args_setting = vars(args)

    # =========================================================================
    u.envs_setting(args_setting['random_seed'])

    # =========================================================================
    # log 생성
    log = lu.get_logger(
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
