'''
install requirement
!pip install numpy
!pip install pandas
!pip install tqdm
!torch install --> https://pytorch.org/get-started/locally/
!pip install mxnet
!pip install gluonnlp
!pip install sentencepiece
!pip install transformers==3.0.2
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
'''

# 일반 module
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

from mylocalmodules import dataloader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils
sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils




def main():
    # -------------------------------------------------------------------------
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing
    parser.add_argument('--root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('--phase', help='현 프로젝트의 진행 단계')
    parser.add_argument('--dataset_name', help='학습에 활용할 데이터셋 이름')
    parser.add_argument('--network', default='kobert', help='학습에 활용할 네트워크 이름')

    # data loader parsing 
    parser.add_argument('--batch_size', type=int, help='배치 사이즈')
    parser.add_argument('--max_len', type=int, help='bert padding 최대 길이')
    parser.add_argument('--shuffle', type=bool, default=False, help='데이터 섞기 여부 / default=False')
    parser.add_argument('--num_workers', type=int, default=1, help='데이터 로딩에 사용하는 subprocess 갯수')
    parser.add_argument('--pin_memory', type=bool, default=False, help='True인 경우 tensor를 cuda 고정 메모리에 올림.')
    parser.add_argument('--drop_last', type=bool, default=False, help='데이터의 마지막 batch 를 사용하지 않음')
    
    # train parsing 
    parser.add_argument('--device_num', type=int, help='사용할 device 번호')
    parser.add_argument('--epochs', type=int, help='학습 에포크')
    parser.add_argument('--pre_trained', help='선행학습 사용 여부')
    parser.add_argument('--optimizer_name', help='생성할 optimizer 이름')
    parser.add_argument('--loss_name', help='생성할 loss function 이름')
    parser.add_argument('--scheduler_name', help='생성할 scheduler 이름')
    parser.add_argument('--learning_rate', type=float, help='학습 learning rate')
    parser.add_argument('--gamma', type=float, default=0.98, help='learning rate 감소 비율 (scheduler 에 따라 다르게 적용)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup scheduler 용 변수')
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--max_grad_norm', type=int, default=1, help='그래디언트 클리핑 기울기')
    
    # re-train parser 
    parser.add_argument('--retrain', type=bool, default=False) # sh 파일에서 설정 안됨.
    parser.add_argument('--trained_weight')
    parser.add_argument('--start_epoch', type=int)
    
    # envs parsing 
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # get save path
    model_save_path = utils.get_save_path(args)

    # envs setting 
    utils.envs_setting(args.random_seed)

    # log
    os.makedirs(f'./log', exist_ok=True)
    log = utils.get_logger(
        get='TRAIN', 
        log_file_name=f'./log/train.log', 
        time_handler=True, 
        console_display=False, 
        logging_level='info'
    )
    
    # -------------------------------------------------------------------------
    # data
    log.info('>>> read data')
    with open(f'{args.root_path}/datasets/{args.dataset_name}/train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(f'{args.root_path}/datasets/{args.dataset_name}/val_data.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    # json_dataset = dutils.make_json_dataset(load_path=f'{args.root_path}/datasets/{args.dataset_name}')
    
    log.info('>>> class_dict')
    with open(f'{args.root_path}/datasets/{args.dataset_name}/dataset_info.json', 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    class_dict = dataset_info['class_dict']
    uni_class_list = list(class_dict.keys())

    log.info('>>> get kobert tokenizer')
    tokenizer = dutils.get_kobert_tokenizer(pre_trained=args.pre_trained)
    
    log.info('>>> get vocab')
    vocab = dutils.get_vocab(tokenizer)

    log.info('>>> Train dataloader 생성')

    Train_Dataloader = dutils.get_dataloader(
        string_list=train_data['string'][:100],
        label_list=train_data['label'][:100],
        batch_size=args.batch_size,
        tokenizer=tokenizer.tokenize,
        vocab=vocab,
        max_len=args.max_len,
        pad=True,
        pair=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )

    log.info('>>> Validation dataloader 생성')
    Val_Dataloader = dutils.get_dataloader(
        string_list=val_data['string'][:100],
        label_list=val_data['label'][:100],
        batch_size=args.batch_size,
        tokenizer=tokenizer.tokenize,
        vocab=vocab, 
        max_len=args.max_len,
        pad=True,
        pair=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )
    
    # -------------------------------------------------------------------------
    # model
    device = tutils.get_device(args.device_num)

    log.info('>>> get kobert pretrained model')
    model = mutils.get_bert_model(
        method='BertForSequenceClassification', 
        pre_trained="skt/kobert-base-v1", 
        num_labels=len(class_dict)
    )
    model = model.to(device)

    # -------------------------------------------------------------------------
    # optimizer, loss function, scheduler
    optimizer = tutils.get_optimizer(
        base='transformers', 
        method=args.optimizer_name, 
        model=model, 
        learning_rate=args.learning_rate
    )
    loss_function = tutils.get_loss_function(method=args.loss_name)
    scheduler = tutils.get_scheduler(
        base='transformers',
        method=args.scheduler_name,
        optimizer=optimizer,
        t_total=len(Train_Dataloader) * args.epochs,
        warmup_ratio=args.warmup_ratio
    )

    # -------------------------------------------------------------------------
    # train
    if args.amp:
        log.info('>>> amp 적용')
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        
    if args.retrain:
        log.info(f'>>> load weigth: {args.root_path}/{args.trained_weight}')
        model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
    else:
        args.start_epoch = 1

    log.info('>>> train')
    tutils.train(
        model=model,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        train_dataloader=Train_Dataloader,
        validation_dataloader=Val_Dataloader,
        class_list=uni_class_list,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,        
        model_save_path=model_save_path
    )

    # -------------------------------------------------------------------------
    log.info('>>> json 데이터 저장')
    with open(f'{model_save_path}/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent='\t', ensure_ascii=False)
    


if __name__ == '__main__':
    main()