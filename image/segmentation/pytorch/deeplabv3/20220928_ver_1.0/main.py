# common module
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json

# custom module
from mylocalmodules import data_loader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils

sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils


def main():
    # -------------------------------------------------------------------------
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing 
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('dataset_name', help='학습에 활용할 데이터셋 이름')
    parser.add_argument('--network', help='학습에 활용할 네트워크 이름')
    
    # data loader parsing
    parser.add_argument('class_list', help='학습시 분류할 클래스 리스트')
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

    # -------------------------------------------------------------------------
    # parse info save
    model_save_path = utils.get_save_path(args)

    # environment setting
    utils.envs_setting(args.random_seed)

    # make logger
    logger = utils.get_logger('train', logging_level='info')

    # -------------------------------------------------------------------------
    # class_dict
    class_list = args.class_list.split(',')
    class_dict = utils.make_class_dict(class_list)

    # # color_dict
    # with open(f'{args.root_path}/color_dict.json', 'r', encoding='utf-8') as f:
    #     color_dict = json.load(f)

    # get dataloader
    print('\n>>> get dataloader')
    Train_Dataloader = dutils.get_dataloader(mode='train',
                                             img_path=f'{args.root_path}/datasets/{args.dataset_name}/train/images',
                                             label_path=f'{args.root_path}/datasets/{args.dataset_name}/train/label',
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory,
                                             drop_last=args.drop_last)
    
    Val_Dataloader = dutils.get_dataloader(mode='val',
                                           img_path=f'{args.root_path}/datasets/{args.dataset_name}/val/images',
                                           label_path=f'{args.root_path}/datasets/{args.dataset_name}/val/label',
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory,
                                           drop_last=args.drop_last)
    
    # -------------------------------------------------------------------------
    # device selection
    device = tutils.get_device(args.device_num)

    # model
    print('\n>>> get model...')
    model = mutils.GetModel(model_name=args.network,
                            pretrained=False, 
                            n_outputs=len(class_list))
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
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # select retrain
    if args.retrain:
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
        print('\n>>> re-training')
    else:
        args.start_epoch = 1
        print('\n>>> training...')

    # train 
    print('\n>>> training...')
    tutils.train(
        model=model,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        # batch_size=args.batch_size,
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
    
    # -------------------------------------------------------------------------
    # best model name 
    # best_model_name = f'batch{args.batch_size}_epoch{str(train_history["best"]["epoch"]).zfill(4)}'

    # # result df save ----------------------------------------------------------
    # val_class_ary = []
    # best_val_pred_class_ary = []
    # for true_label, pred_label in zip(Val_Dataloader.dataset.label_list, best_val_pred_label_ary):
    #     val_class_ary.append(class_list[true_label])
    #     best_val_pred_class_ary.append(class_list[pred_label])
    
    # result_df = pd.DataFrame([val_class_ary, best_val_pred_class_ary, Val_Dataloader.dataset.imgae_name_list], index=['true', 'pred', 'name']).T
    # utils.save_csv(save_path=f'{model_save_path}/{best_model_name}/result_df.csv', data_for_save=result_df, index=False)




if __name__ == '__main__':
    main()
