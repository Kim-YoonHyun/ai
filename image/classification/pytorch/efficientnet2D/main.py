'''
install requirement
!pip install timm
'''

# common module
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
sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils
from mylocalmodules import data_loader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils



def main():
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('dataset_name', help='학습에 활용할 데이터셋 이름')
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
    # get save path
    model_save_path = utils.get_save_path(args)

    utils.envs_setting(args.random_seed)

    # make logger
    logger = utils.get_logger(f'{args.root_path}/train', logging_level='info')

    # -------------------------------------------------------------------------
    # class_dict
    with open(f'{args.root_path}/datasets/{args.dataset_name}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)
    
    class_dict = dataset_info['class_info']['dict']
    class_list = list(class_dict.keys())

    with open(f'{model_save_path}/dataset_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(dataset_info, f, indent='\t', ensure_ascii=False)
    # -------------------------------------------------------------------------
    # annotation
    train_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/train/annotation.csv')
    train_label_list = []
    for train_class_name in train_annotation_df['class']:
        train_label_list.append(class_dict[train_class_name])
    train_annotation_df['label'] = train_label_list
    logger.info(train_annotation_df)
    
    val_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/val/annotation.csv')
    val_label_list = []
    for val_class_name in val_annotation_df['class']:
        val_label_list.append(class_dict[val_class_name])
    val_annotation_df['label'] = val_label_list

    # -------------------------------------------------------------------------
    # get dataloader
    print('\n>>> get dataloader')
    Train_Dataloader = dutils.get_dataloader(mode='train',
                                             img_path=f'{args.root_path}/datasets/{args.dataset_name}/train/images',
                                             annotation_df=train_annotation_df,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory,
                                             drop_last=args.drop_last)

    Val_Dataloader = dutils.get_dataloader(mode='val',
                                           img_path=f'{args.root_path}/datasets/{args.dataset_name}/val/images',
                                           annotation_df=val_annotation_df,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory,
                                           drop_last=args.drop_last)
    
    # -------------------------------------------------------------------------
    # GPU
    tutils.see_device(logger)
    device = tutils.get_device(args.device_num)

    # model
    print('\n>>> get pretrained model...')
    model = mutils.GetModel(model_name=args.network, pretrained=True, n_outputs=len(class_list))
    model = model.to(device)

    # -------------------------------------------------------------------------
    # optimizer, loss function, scheduler
    optimizer = tutils.get_optimizer(base='torch', method=args.optimizer_name, model=model, learning_rate=args.learning_rate)
    loss_function = tutils.get_loss_function(method=args.loss_name)
    scheduler = tutils.get_scheduler(base='torch',
                                     method=args.scheduler_name,
                                     optimizer=optimizer,
                                     t_total=len(Train_Dataloader) * args.epochs,
                                     warmup_ratio=args.warmup_ratio)
    # amp
    if args.amp:
        from apex import amp
        '''
        mixed precision training

        처리 속도를 높이기 위한 FP16(16bit floating point)연산과 정확도 유지를 위한 FP32 연산을 섞어 학습하는 방법
        Tensor Core를 활용한 FP16연산을 이용하면 FP32연산 대비 절반의 메모리 사용량과 8배의 연산 처리량 & 2배의 메모리 처리량 효과가 있다
        '''
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # select retrain
    if args.retrain:
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
        print('\n>>> re-training')
    else:
        args.start_epoch = 1
        print('\n>>> training...')
    
    # train
    # model_save_path = f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}'
    tutils.train(
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
        model_save_path=model_save_path)


if __name__ == '__main__':
    main()
            
