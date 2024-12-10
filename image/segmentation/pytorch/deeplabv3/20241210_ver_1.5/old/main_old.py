'''
pip install transformers
'''

# common module
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
import time

# custom module
from mylocalmodules import dataloader as dm
from mylocalmodules import train as tm
from mylocalmodules import model as mm

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils
from sharemodule import trainutils as tutils


def main():
    # =====================================================================
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing 
    parser.add_argument('--root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('--phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('--dataset_name', help='학습에 활용할 데이터셋 이름')
    parser.add_argument('--network', help='학습에 활용할 네트워크 이름')
    
    # data loader parsing
    parser.add_argument('--batch_size', type=int, help='배치 사이즈')
    parser.add_argument('--shuffle', type=bool, default=False, help='데이터 섞기 여부 / default=False')
    parser.add_argument('--num_workers', type=int, default=5, help='데이터 로딩에 사용하는 subprocess 갯수')
    parser.add_argument('--pin_memory', type=bool, default=False, help='True인 경우 tensor를 cuda 고정 메모리에 올림.')
    parser.add_argument('--drop_last', type=bool, default=False, help='데이터의 마지막 batch 를 사용하지 않음')

    # train parsing
    parser.add_argument('--device_num', type=int, help='사용할 device 번호')
    parser.add_argument('--epochs', type=int, help='학습 에포크')
    parser.add_argument('--optimizer_name', help='생성할 optimizer 이름')
    parser.add_argument('--loss_name', help='생성할 loss function 이름')
    parser.add_argument('--scheduler_name', help='생성할 scheduler 이름')
    parser.add_argument('--learning_rate', type=float, help='학습 learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate 감소 비율 (scheduler 에 따라 다르게 적용)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup scheduler 용 변수')
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=int, default=1, help='그래디언트 클리핑 기울기')
    
    # re-train parser
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--trained_weight')
    parser.add_argument('--start_epoch', type=int)

    # envs parsing
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    # =====================================================================
    start = time.time()

    # parse info save
    model_save_path = utils.get_save_path(args)

    # environment setting
    utils.envs_setting(args.random_seed)

    # make logger
    os.makedirs(f'./log', exist_ok=True)
    log = utils.get_logger(
        get='TRAIN', 
        log_file_name=f'./log/train.log', 
        time_handler=True, 
        console_display=False, 
        logging_level='info'
    )

    # =====================================================================
    log.info('>>> dataset_info 불러오기')
    with open(f'{args.root_path}/datasets/{args.dataset_name}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)
    class_dict = dataset_info['class']['dict']
    uni_class_list = list(class_dict.keys())
    
    # color_dict 
    color_dict = dataset_info['color']['dict']

    class_info = {}
    class_info['class_dict'] = class_dict
    class_info['color_dict'] = color_dict
    log.info('>>> class_info 저장')
    with open(f'{model_save_path}/class_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(class_info, f, indent='\t', ensure_ascii=False)

    # =====================================================================
    # train 이터레이터 생성
    Train_Dataloader = dm.get_dataloader(
        img_path=f'{args.root_path}/datasets/{args.dataset_name}/train/images',
        label_path=f'{args.root_path}/datasets/{args.dataset_name}/train/label',
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )
    Train_Iterator = tm.Iterator(Train_Dataloader, model, device)

    # =====================================================================
    # validation 이터레이터 생성
    Val_Dataloader = dm.get_dataloader(
        img_path=f'{args.root_path}/datasets/{args.dataset_name}/val/images',
        label_path=f'{args.root_path}/datasets/{args.dataset_name}/val/label',
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )
    Val_Iterator = tm.Iterator(Val_Dataloader, model, device)

    # =====================================================================
    # log.info('>>> gpu 선택')
    device = tutils.get_device(args.device_num)
    # log.info(f'device: {device}')

    # log.info('>>> model 불러오기')
    model = mm.GetModel(
        model_name=args.network,
        pretrained=False, 
        n_outputs=len(uni_class_list)
    )
    model = model.to(device)

    # =====================================================================
    # optimizer & loss function & scheduler
    # log.info('>>> optimizer 생성')
    optimizer = tutils.get_optimizer(
        base='torch',
        method=args.optimizer_name, 
        model=model, 
        learning_rate=args.learning_rate
    )

    log.info('>>> loss function 생성')
    loss_function = tutils.get_loss_function(method=args.loss_name)

    log.info('>>> scheduler 생성')
    scheduler = tutils.get_scheduler(
        base='torch',
        method=args.scheduler_name,
        optimizer=optimizer,
        t_total=len(Train_Dataloader) * args.epochs,
        warmup_ratio=args.warmup_ratio
    )
    # =====================================================================
    if args.amp:
        log.info('>>> amp 적용')
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.retrain:
        log.info('>>> 재학습 적용')
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
        print('\n>>> re-training')
    else:
        args.start_epoch = 0

    # train 
    log.info('>>> 학습 시작')
    tutils.train(
        model=model,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        train_iterator=Train_Iterator,
        validation_iterator=Val_Iterator,
        train_dataloader=Train_Dataloader,
        validation_dataloader=Val_Dataloader,
        uni_class_list=uni_class_list,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        reset_class=True,
        model_save_path=model_save_path
    )

    h, m, s = utils.time_measure(start)
    log.info(f'>>> {args.dataset_name} 학습완료. 소요시간: {h}시간 {m}분 {s}초')
    

if __name__ == '__main__':
    main()
