'''
sharemodule ver 2.5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install psutil
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
    parser.add_argument('--root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('--root_dataset_path')
    parser.add_argument('--phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('--dataset_name', help='학습에 활용할 데이터셋 이름')
    parser.add_argument('--network', help='학습에 활용할 네트워크 이름')
    parser.add_argument('--purpose')
    
    # train parameter
    parser.add_argument('--device_num', type=int, help='사용할 device 번호')
    parser.add_argument('--epochs', type=int, help='학습 에포크')
    parser.add_argument('--batch_size', type=int, help='배치 사이즈')
    parser.add_argument('--max_grad_norm', type=int, default=1, help='그래디언트 클리핑 기울기')
    parser.add_argument('--loss_function_name', help='생성할 loss function 이름')
    parser.add_argument('--optimizer_name', help='생성할 optimizer 이름')
    parser.add_argument('--scheduler_name', help='생성할 scheduler 이름')
    parser.add_argument('--gamma', type=float, help='learning rate 감소 비율 (scheduler 에 따라 다르게 적용)')
    parser.add_argument('--learning_rate', type=float, help='학습 learning rate')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--shuffle', type=bool, default=False, help='데이터 섞기 여부 / default=False')
    parser.add_argument('--drop_last', type=bool, default=False, help='데이터의 마지막 batch 를 사용하지 않음')
    parser.add_argument('--num_workers', type=int, default=5, help='데이터 로딩에 사용하는 subprocess 갯수')
    parser.add_argument('--pin_memory', type=bool, default=False, help='True인 경우 tensor를 cuda 고정 메모리에 올림.')
    parser.add_argument('--pre_trained', type=str, default='None')
    parser.add_argument('--start_epoch', type=int)
    
    args = parser.parse_args()
    
    return args

    
def trainer(args_setting):
    # =========================================================================
    # 데이터 셋 경로
    dataset_path = f"{args_setting['root_dataset_path']}/datasets/{args_setting['dataset_name']}"
    
    with open(f'{dataset_path}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)
    class_dict = dataset_info['class']['dict']
    #==
    class_list = list(class_dict.keys())
    id_list = list(class_dict.values())
    id2label_dict = dict(zip(id_list, class_list))
    #==
    uni_class_list = list(class_dict.keys())
    
    # color_dict 
    color_dict = dataset_info['color']['dict']

    class_info = {}
    class_info['class_dict'] = class_dict
    class_info['color_dict'] = color_dict
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dl.get_dataloader(
        img_path=f'{dataset_path}/train/images',
        label_path=f'{dataset_path}/train/label',
        batch_size=args_setting['batch_size'],
        shuffle=args_setting['shuffle'],
        num_workers=args_setting['num_workers'],
        pin_memory=args_setting['pin_memory'],
        drop_last=args_setting['drop_last']
    )
    
    print('validation dataloader 생성 중...')
    Val_Dataloader = dl.get_dataloader(
        img_path=f'{dataset_path}/val/images',
        label_path=f'{dataset_path}/val/label',
        batch_size=args_setting['batch_size'],
        shuffle=args_setting['shuffle'],
        num_workers=args_setting['num_workers'],
        pin_memory=args_setting['pin_memory'],
        drop_last=args_setting['drop_last']
    )
    
    # =========================================================================
    # device 
    tru.see_device()
    device = tru.get_device(args_setting['device_num'])
    
    # =========================================================================
    # model        
    model = net.GetModel(
        model_name=args_setting['network'],
        pretrained=False, 
        n_outputs=len(uni_class_list)
    )
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
        optimizer=optimizer,
        gamma=args_setting['gamma']
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
    
    # 데이터 정보 저장
    with open(f'{model_save_path}/class_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(class_info, f, indent='\t', ensure_ascii=False)
        
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
        id2label_dict=id2label_dict
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
