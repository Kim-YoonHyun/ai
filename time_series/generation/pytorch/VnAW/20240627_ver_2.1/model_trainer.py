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
from model_origin_3 import network as net

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
    parser.add_argument('--reduce_num', type=int, default=0)
    parser.add_argument('--scale', type=str, default='fixed-max')
    parser.add_argument('--x_p', type=int)
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
    
    # network parameter
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=1064, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--embed_type', type=str)
    parser.add_argument('--temporal_type', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--projection_type', type=str)

    # layer num parameter
    parser.add_argument('--enc_layer_num', type=int, help='num of encoder layers')
    parser.add_argument('--dec_layer_num', type=int, help='num of decoder layers')
    
    # activation parameter
    parser.add_argument('--enc_activation', type=str, default='relu', help='activation')
    parser.add_argument('--dec_activation', type=str, default='gelu', help='activation')
    
    args = parser.parse_args()
    return args


def temporal_type_setting(args, args_setting):
    if args.embed_type == 'pure':
        args_setting['temporal_type'] = None
    return args_setting
    
    
def activation_setting(args, args_setting):
    # activation
    if args.enc_layer_num == 0:
        args_setting['enc_activation'] = None
    if args.dec_layer_num == 0:
        args_setting['dec_activation'] = None
    return args_setting


def reducing(data_list, total_num, reduce_num=0):
    if reduce_num == 0:
        size = total_num
    else:
        size = reduce_num
    filter_list = np.array(data_list)[:size].tolist()
    return filter_list


def length_setting(dataloader, args_setting):
    temp_dataloader = copy.deepcopy(dataloader)
    x, mark, _, y, _, _, _ = next(iter(temp_dataloader))
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    mark = mark.cpu().detach().numpy()
    
    x_len = x.shape[1]
    y_len = y.shape[1]
    enc_feature_len = x.shape[-1]
    dec_feature_len = y.shape[-1]
    enc_temporal_len = mark.shape[-1]
    dec_temporal_len = mark.shape[-1]
    out_feature_len = dec_feature_len
    
    args_setting['x_len'] = x_len
    args_setting['y_len'] = y_len
    args_setting['enc_feature_len'] = enc_feature_len
    args_setting['dec_feature_len'] = dec_feature_len
    if args_setting['embed_type'] == 'pure':
        args_setting['enc_temporal_len'] = None
        args_setting['dec_temporal_len'] = None
    else:
        args_setting['enc_temporal_len'] = enc_temporal_len
        args_setting['dec_temporal_len'] = dec_temporal_len
    args_setting['out_feature_len'] = out_feature_len
    return args_setting
    
    
def trainer(args_setting):
    
    # =========================================================================
    # 데이터 셋 경로
    dataset_path = f"{args_setting['root_dataset_path']}/datasets/{args_setting['dataset_name']}"
    
    print('데이터 이름 리스트 dict 생성')
    name_dict = {}
    for tv in ['train', 'val']:
        name_dict[tv] = {}
        for xym in ['x', 'y', 'mark']:
            name_list = os.listdir(f'{dataset_path}/{tv}/{xym}')
            name_list.sort()
            total_num = len(name_list)
            filter_name_list = reducing(
                data_list=name_list, 
                total_num=total_num, 
                reduce_num=args_setting['reduce_num']
            )
            name_dict[tv][xym] = filter_name_list
            
    # norm 불러오기
    norm_path = '/'.join(dataset_path.split('/')[:-2])
    with open(f'{norm_path}/norm.json', 'r', encoding='utf-8-sig') as f:
        norm = json.load(f)
    with open(f'{norm_path}/info.json', 'r', encoding='utf-8-sig') as f:
        info = json.load(f)
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dam.get_dataloader(
        mode='train',
        n_heads=args_setting['n_heads'],
        x_p=args_setting['x_p'],
        batch_size=args_setting['batch_size'], 
        dataset_path=f'{dataset_path}/train', 
        x_name_list=name_dict['train']['x'], 
        y_name_list=name_dict['train']['y'], 
        mark_name_list=name_dict['train']['mark'], 
        scale=args_setting['scale'],
        norm=norm,
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'],
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    print('validation dataloader 생성 중...')
    Val_Dataloader = dam.get_dataloader(
        mode='train',
        n_heads=args_setting['n_heads'],
        x_p=args_setting['x_p'],
        batch_size=args_setting['batch_size'], 
        dataset_path=f'{dataset_path}/val', 
        x_name_list=name_dict['val']['x'], 
        y_name_list=name_dict['val']['y'], 
        mark_name_list=name_dict['val']['mark'], 
        scale=args_setting['scale'],
        norm=norm,
        shuffle=args_setting['shuffle'], 
        drop_last=args_setting['drop_last'], 
        num_workers=args_setting['num_workers'], 
        pin_memory=args_setting['pin_memory'], 
        sampler_name=args_setting['sampler_name']
    )
    
    # =========================================================================
    # 출력 length 세팅
    new_args_setting = length_setting(
        dataloader=Train_Dataloader,
        args_setting=args_setting
    )
    
    # =========================================================================
    # device 
    tum.see_device()
    device = tum.get_device(new_args_setting['device_num'])
    
    # =========================================================================
    # model        
    model = net.VnAW(
        x_len=new_args_setting['x_len'],
        y_len=new_args_setting['y_len'],
        embed_type=new_args_setting['embed_type'],
        d_model=new_args_setting['d_model'],
        d_ff=new_args_setting['d_ff'],
        n_heads=new_args_setting['n_heads'],
        projection_type=new_args_setting['projection_type'],
        temporal_type=new_args_setting['temporal_type'],
        enc_layer_num=new_args_setting['enc_layer_num'],
        dec_layer_num=new_args_setting['dec_layer_num'],
        enc_feature_len=new_args_setting['enc_feature_len'],
        dec_feature_len=new_args_setting['dec_feature_len'],
        enc_temporal_len=new_args_setting['enc_temporal_len'],
        dec_temporal_len=new_args_setting['dec_temporal_len'],
        enc_act=new_args_setting['enc_activation'],
        dec_act=new_args_setting['dec_activation'],
        dropout_p=new_args_setting['dropout_p'],
        output_length=new_args_setting['out_feature_len']
    )
    model.to(device)

    # pre-trained
    torch.autograd.set_detect_anomaly(True)
    if new_args_setting['pre_trained'] != 'None':
        model.load_state_dict(torch.load(f"{new_args_setting['pre_trained']}"))
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
        method=new_args_setting['optimizer_name'],
        model=model,
        learning_rate=new_args_setting['learning_rate']
    )

    # loss function
    loss_function = lfm.LossFunction(
        base='torch',
        method=new_args_setting['loss_function_name']
    )
    # scheduler
    total_iter = new_args_setting['epochs'] * (1+len(name_dict['train']['x']) // new_args_setting['batch_size'])
    warmup_iter = int(total_iter * 0.1)
    scheduler = tum.get_scheduler(
        base='torch',
        method=new_args_setting['scheduler_name'],
        optimizer=optimizer,
        total_iter=total_iter,
        warmup_iter=warmup_iter
    )
    
    # =========================================================================
    # save setting
    root_save_path = f"{new_args_setting['root_path']}/trained_model/{new_args_setting['phase']}/{new_args_setting['dataset_name']}"
    condition_order = tum.get_condition_order(
        args_setting=new_args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs', 'device_num', 'batch_size']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(new_args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose=new_args_setting['purpose'], 
        start_epoch=0, 
        epochs=new_args_setting['epochs'], 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=new_args_setting['max_grad_norm'], 
        reset_class=None, 
        model_save_path=model_save_path
    )
    print(condition_order)


def main():
    args = get_args()
    args_setting = vars(args)
    args_setting = temporal_type_setting(args, args_setting)
    args_setting = activation_setting(args, args_setting)

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