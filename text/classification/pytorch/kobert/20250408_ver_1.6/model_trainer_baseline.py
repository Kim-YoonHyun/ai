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

    
def trainer(args_setting):
    # =========================================================================
    def get_model(self, device, class_num, weight=None):
        model = mutils.get_bert_model(method='BertForSequenceClassification', pre_trained="skt/kobert-base-v1", num_labels=class_num)
        if weight:
            if device == torch.device('cpu'):
                model.load_state_dict(torch.load(weight, map_location=device))
            else:
                model.load_state_dict(torch.load(weight))
        model = model.to(device)
        return model
    
    model = get_model(device, class_num=5)
    sys.exit()
    
    # 데이터 셋 경로
    dataset_path = f"{args_setting['root_dataset_path']}/datasets/{args_setting['dataset_name']}"
    
    '''
    dataset code
    '''
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dl.get_dataloader(
        mode='train',
        batch_size=args_setting['batch_size'],
        dataset_path=f'{dataset_path}/train',
        x_name_list=None,
        y_name_list=None,
        scale=args_setting['scale'],
        norm=None,
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'],
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    print('validation dataloader 생성 중...')
    Val_Dataloader = dl.get_dataloader(
        mode='train',
        batch_size=args_setting['batch_size'], 
        dataset_path=f'{dataset_path}/val', 
        x_name_list=None, 
        y_name_list=None, 
        scale=args_setting['scale'],
        norm=None,
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'], 
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    
    # =========================================================================
    # device 
    tru.see_device()
    device = tru.get_device(args_setting['device_num'])
    
    # =========================================================================
    # model        
    model = net.network()
    model.to(device)

    # pre-trained
    start_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    if args_setting['pre_trained'] != 'None':
        model.load_state_dict(torch.load(f"{args_setting['pre_trained']}"))
        print('\n>>> use pre-trained model')
        start_epoch = 0
        
    # 파라미터 갯수 출력
    tru.see_parameters(model)
    
    # =========================================================================
    # optimizer
    optimizer = tru.get_optimizer(
        base='torch',
        method=args_setting['optimizer_name'],
        model=model,
        learning_rate=args_setting['learning_rate']
    )

    # loss function
    criterion = lf.LossFunction(
        base='torch',
        method=args_setting['loss_function_name']
    )
    # scheduler
    scheduler = tru.get_scheduler(
        base='torch',
        method=args_setting['scheduler_name'],
        optimizer=optimizer
    )
    
    # =========================================================================
    # save setting
    root_save_path = f"{args_setting['root_path']}/trained_model/{args_setting['phase']}/{args_setting['dataset_name']}"
    condition_order = tru.get_condition_order(
        args_setting=args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs', 'batch_size']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = tr.train(
        model=model, 
        purpose=args_setting['purpose'], 
        start_epoch=start_epoch, 
        epochs=args_setting['epochs'], 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        device=device, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=args_setting['max_grad_norm'], 
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
