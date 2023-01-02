'''
설명
common: 공통
image: 이미지 학습 전용
text: 텍스트 학습 전용
'''

'''
image

install requirement
!pip install timm
'''

'''
text

install requirement
!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers==3.0.2
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
'''

'''common'''
# common module
import random
import os
import sys
import copy
import numpy as np
import pandas as pd
import argparse
import time

'''image'''
# torch
from torch.utils.data import DataLoader
import torch

'''text'''
# kobert
from kobert.pytorch_kobert import get_pytorch_kobert_model

'''common'''
# custom module
from mylocalmodules import data_loader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils
from mylocalmodules import utils 


def main():
    # parsing -----------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # path parsing ------------------------------------------------------------
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('dataset_name', help='학습에 활용할 데이터셋 이름')
    # parser.add_argument('--network', default='swin_v2_cr_small_224', help='학습에 활용할 네트워크 이름')
    parser.add_argument('--network', default='effnet', help='학습에 활용할 네트워크 이름')
    
    # data loader parsing -----------------------------------------------------
    parser.add_argument('class_list', help='학습시 분류할 클래스 리스트')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)

    '''image'''
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=False)

    '''text'''
    parser.add_argument('--max_len', type=int, default=128, help='bert padding 최대 길이')

    '''common'''
    # train parsing -----------------------------------------------------------
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss_name', default='CrossEntropyLoss')
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=int, default=1)

    # re-train parser ---------------------------------------------------------
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--trained_weight', default='<weight_path>/weight.pt')
    parser.add_argument('--start_epoch', type=int, default=192)

    # envs parsing ------------------------------------------------------------
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    
    # environment setting -----------------------------------------------------
    utils.envs_setting(args.random_seed)

    # make logger -------------------------------------------------------------
    logger = utils.get_logger('train', logging_level='info')

    # class_dict --------------------------------------------------------------
    class_list = args.class_list.split(',')
    class_dict = utils.make_class_dict(class_list)

    '''image'''
    # annotation --------------------------------------------------------------
    train_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/train/annotation.csv')
    # train_label_list = []
    # for train_class_name in train_annotation_df['class']:
    #     train_label_list.append(class_dict[train_class_name])
    # train_annotation_df['label'] = train_label_list

    val_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/val/annotation.csv')
    # val_label_list = []
    # for val_class_name in val_annotation_df['class']:
    #     val_label_list.append(class_dict[val_class_name])
    # val_annotation_df['label'] = val_label_list

    # Dataset -----------------------------------------------------------------
    print('\n>>> get dataset')
    Train_Dataset = dutils.TrainDataset(img_path=f'{args.root_path}/datasets/{args.dataset_name}/train/images',
                                        annotation_df=train_annotation_df)
    Val_Dataset = dutils.ValDataset(img_path=f'{args.root_path}/datasets/{args.dataset_name}/val/images',
                                    annotation_df=val_annotation_df)

    # DataLoader --------------------------------------------------------------
    train_dataloader = DataLoader(dataset=Train_Dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                pin_memory=args.pin_memory,
                                drop_last=args.drop_last)
    val_dataloader = DataLoader(dataset=Val_Dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                shuffle=False,
                                pin_memory=args.pin_memory,
                                drop_last=args.drop_last)


    '''text'''
    # get BERT model & Vocabulary ---------------------------------------------
    print('\n>>> get kobert pretrained model...')
    bertmodel, vocab = get_pytorch_kobert_model()

    # get tokenizer -----------------------------------------------------------
    print('\n>>> get tokenizer...')
    tokenizer = tutils.get_bert_tokenizer(vocab=vocab)

    # dataset -----------------------------------------------------------------
    json_dataset = dutils.make_json_dataset(load_path=f'{args.root_path}/datasets/{args.dataset_name}',
                                           class_dict=class_dict)

    # make Json Dataset class -------------------------------------------------
    Train_Dataset = dutils.get_dataset(string_list=json_dataset['train_data']['string'],
                                       label_list=json_dataset['train_data']['label'],
                                       tokenizer=tokenizer,
                                       max_len=args.max_len)
    Val_Dataset = dutils.get_dataset(string_list=json_dataset['validation_data']['string'],
                                       label_list=json_dataset['validation_data']['label'],
                                       tokenizer=tokenizer,
                                       max_len=args.max_len)
    Test_Dataset = dutils.get_dataset(string_list=json_dataset['test_data']['string'],
                                       label_list=json_dataset['test_data']['label'],
                                       tokenizer=tokenizer,
                                       max_len=args.max_len)

    # get data loader ---------------------------------------------------------
    print('\n>>> make data loader')
    Train_Dataloader = dutils.get_dataloader(
        dataset=Train_Dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )
    Validation_Dataloader = dutils.get_dataloader(
        dataset=Val_Dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )
    Test_Dataloader = dutils.get_dataloader(
        dataset=Test_Dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )

    '''common'''
    # device selection --------------------------------------------------------
    device = utils.get_device(args.device_num)

    # model -------------------------------------------------------------------    
    '''image'''
    model_args = {'n_outputs':len(class_list)}
    model = mutils.get_model(model_name=args.network,
                             model_args=model_args)
    '''text'''
    model = mutils.BERTClassifier(bert=bertmodel,
                                  num_classes=len(class_list),
                                  dr_rate=0.5)
    
    '''common'''
    model = model.to(device)
    
    # optimizer, loss function, scheduler -------------------------------------
    optimizer = tutils.get_optimizer(optimizer_name=args.optimizer, model=model, learning_rate=args.learning_rate)
    loss_function = tutils.get_loss_function(loss_name=args.loss_name)
    scheduler = tutils.get_scheduler(method='ExponentialLR', optimizer=optimizer, gamma=args.gamma)
    
    # amp ---------------------------------------------------------------------
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # select retrain ----------------------------------------------------------
    if args.retrain:
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
        print('\n>>> re-training')
    else:
        args.start_epoch = 1
        print('\n>>> training...')
    
    # train -------------------------------------------------------------------
    print('\n>>> training...')
    model_save_path = f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}'
    best_val_pred_label_ary, trained_model, train_history = tutils.train(
        model=model,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        class_list=class_list,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        model_save_path=model_save_path)


    # best model name ---------------------------------------------------------
    best_model_name = f'batch{args.batch_size}_epoch{str(train_history["best"]["epoch"]).zfill(4)}'

    '''text'''
    # dataset save ------------------------------------------------------------
    utils.save_json(save_path=f'{model_save_path}/{best_model_name}/{args.dataset_name.split(".")[0]}.json', data_for_save=json_dataset)

    '''common'''
    # result df save ----------------------------------------------------------
    val_class_ary = []
    best_val_pred_class_ary = []
    for true_label, pred_label in zip(Val_Dataset.label_list, best_val_pred_label_ary):
        val_class_ary.append(class_list[true_label])
        best_val_pred_class_ary.append(class_list[pred_label])
        
    result_df = pd.DataFrame([val_class_ary, best_val_pred_class_ary, val_annotation_df['name'].values], index=['true', 'pred', 'name']).T
    utils.save_csv(save_path=f'{model_save_path}/{}/result_df.csv', data_for_save=result_df, index=False)


if __name__ == '__main__':
    main()
            
