# common module
import time
import argparse
import os
import sys
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# custom module
from mylocalmodules import data_loader as dm
from mylocalmodules import model as mm
from mylocalmodules import train as tm

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils
from sharemodule import trainutils as tutils


def main():
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing
    parser.add_argument('--root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('--phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('--dataset_name', help='학습에 활용할 데이터셋 이름')
    # parser.add_argument('--network', default='swin_v2_cr_small_224', help='학습에 활용할 네트워크 이름')
    parser.add_argument('--network', default='efficientnet_b4', help='학습에 활용할 네트워크 이름')
    
    # data loader parsing
    parser.add_argument('--batch_size', type=int, default=16, help='배치 사이즈')
    parser.add_argument('--shuffle', type=bool, default=False, help='데이터 섞기 여부 / default=False')
    parser.add_argument('--num_workers', type=int, default=1, help='데이터 로딩에 사용하는 subprocess 갯수')
    parser.add_argument('--pin_memory', type=bool, default=True, help='True인 경우 tensor를 cuda 고정 메모리에 올림.')
    parser.add_argument('--drop_last', type=bool, default=False, help='데이터의 마지막 batch 를 사용하지 않음')

    # train parsing 
    parser.add_argument('--device_num', type=int, default=0, help='사용할 device 번호')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에포크')
    parser.add_argument('--optimizer_name', default='AdamW', help='생성할 optimizer 이름')
    parser.add_argument('--loss_name', default='CrossEntropyLoss', help='생성할 loss function 이름')
    parser.add_argument('--scheduler_name', default='ExponentialLR', help='생성할 scheduler 이름')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='학습 learning rate')
    parser.add_argument('--gamma', type=float, default=0.98, help='learning rate 감소 비율 (scheduler 에 따라 다르게 적용)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup scheduler 용 변수')
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=int, default=1, help='그래디언트 클리핑 기울기')

    # re-train parser
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--trained_weight', default='1st_phase/beef_hsv_4part_model/effnet/batch16_epoch0191/weight.pt')
    parser.add_argument('--start_epoch', type=int, default=192)

    # envs parsing
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    start = time.time()

    # get save path
    model_save_path = utils.get_save_path(args)

    utils.envs_setting(args.random_seed)

    # make logger
    os.makedirs('./log', exist_ok=True)
    log = utils.get_logger(
        get='TRAIN', 
        log_file_name='./log/train.log', 
        time_handler=True, 
        console_display=False, 
        logging_level='info'
    )
    # -------------------------------------------------------------------------
    log.info('>>> dataset_info 불러오기')
    with open(f'{args.root_path}/datasets/{args.dataset_name}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)
    class_dict = dataset_info['class']['dict']
    class_list = list(class_dict.keys())

    with open(f'{model_save_path}/dataset_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(dataset_info, f, indent='\t', ensure_ascii=False)

    # -------------------------------------------------------------------------
    log.info('>>> annotation 불러오기')
    train_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/train/annotation.csv')
    train_label_list = []
    for train_class_name in train_annotation_df['class']:
        train_label_list.append(class_dict[train_class_name])
    train_annotation_df['label'] = train_label_list
    
    val_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/val/annotation.csv')
    val_label_list = []
    for val_class_name in val_annotation_df['class']:
        val_label_list.append(class_dict[val_class_name])
    val_annotation_df['label'] = val_label_list

    # -------------------------------------------------------------------------
    # get dataloader
    log.info('dataloader 생성')
    Train_Dataloader = dm.get_dataloader(
        mode='train',
        img_path=f'{args.root_path}/datasets/{args.dataset_name}/train/images',
        annotation_df=train_annotation_df,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )

    Val_Dataloader = dm.get_dataloader(
        mode='val',
        img_path=f'{args.root_path}/datasets/{args.dataset_name}/val/images',
        annotation_df=val_annotation_df,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )
    
    # -------------------------------------------------------------------------
    log.info('>>> device 선택')
    device = tutils.get_device(args.device_num)
    log.info(f'device: {device}')

    # model
    log.info('>>> model 불러오기')
    model = mm.GetModel(
        model_name=args.network, 
        pretrained=True, 
        n_outputs=len(class_list)
    )
    model = model.to(device)

    # -------------------------------------------------------------------------
    log.info('>>> optimizer 생성')
    optimizer = tutils.get_optimizer(
        base='torch', 
        method=args.optimizer_name, 
        model=model, 
        learning_rate=args.learning_rate
    )
    log.info('>>> log function 생성')
    loss_function = tutils.get_loss_function(method=args.loss_name)
    log.info('>>> scheduler 생성')
    scheduler = tutils.get_scheduler(
        base='torch',
        method=args.scheduler_name,
        optimizer=optimizer,
        t_total=len(Train_Dataloader) * args.epochs,
        warmup_ratio=args.warmup_ratio
    )
    # -------------------------------------------------------------------------
    if args.amp:
        log.info('>>> amp 적용')
        from apex import amp
        '''
        mixed precision training

        처리 속도를 높이기 위한 FP16(16bit floating point)연산과 정확도 유지를 위한 FP32 연산을 섞어 학습하는 방법
        Tensor Core를 활용한 FP16연산을 이용하면 FP32연산 대비 절반의 메모리 사용량과 8배의 연산 처리량 & 2배의 메모리 처리량 효과가 있다
        '''
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.retrain:
        log.info('>>> 재학습')
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
        print('\n>>> re-training')
    else:
        args.start_epoch = 1
    
    # train
    log.info('>>> 학습 시작')
    tm.train(
        model=model,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        train_dataloader=Train_Dataloader,
        validation_dataloader=Val_Dataloader,
        class_list=class_list,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        reset_class=False,
        model_save_path=model_save_path
    )

    h, m, s = utils.time_measure(start)
    log.info(f'>>> 학습완료. 소요시간: {h}시간 {m}분 {s}초')


if __name__ == '__main__':
    main()
            
