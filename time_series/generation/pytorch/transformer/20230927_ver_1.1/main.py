'''
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install matplotlib
pip install tqdm
pip install -U scikit-learn
'''

import sys
import os

# import shutil
import time
import json
import torch
import argparse
import numpy as np
import random
import pandas as pd
# from tqdm import tqdm

# import copy

# from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from models import Transformer
from utils.warm_up import LearningRateWarmUP

from mylocalmodules import dataloader as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm


def basic_set():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def parameter_setup(args, feature_length, condition_num):
    args.enc_in = feature_length
    args.dec_in = feature_length
    args.c_out = feature_length
    args.freq = condition_num
    return args


def set_device():
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size


def get_device(gpu_idx):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument("--group", type=str)
    parser.add_argument('--dataset_name')
    parser.add_argument("--column_type", type=str, help = "S : single, M : multi")
    parser.add_argument('--train_p', type=float)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--ddp", type=str, default="non")

    # train variable
    parser.add_argument('--purpose', type=str)
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--optimizer_name', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--total_iter', type=float, default=100)
    parser.add_argument('--warmup_iter', type=float, default=10)
    parser.add_argument('--retrain', type=bool, default=False)

    # transformer embedding
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')

    # transformer common
    parser.add_argument('--pred_len', type=int, default=600, help='prediction sequence length')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=1064, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # transformer encoder
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')

    # transformer decoder
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    # Formers 
    parser.add_argument('--seq_len', type=int, default=600, help='input sequence length')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    args = parser.parse_args()
    args_setting = vars(args)
    # =========================================================================
    basic_set()
    root_save_path = f'{args.root_path}/{args.group}/trained_model'

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=args.root_path,
        log_file_name=f'train_{args.group}.log',
        time_handler=True
    )
    whole_start = time.time()

    # =========================================================================
    # 컬럼별 진행
    col_list = os.listdir(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}')
    for col in col_list:
        # ==
        # if train_col != 'ECU_InjectionGasTemp':
        #     continue
        # ==
        col_start = time.time()
        log.info(f'\n{args.group}, {args.column_type}, {col}')
        print(args.group, args.column_type, col)

        # =========================================================================
        # data 불러오기
        print('data loading...')
        tar_train_name_list = os.listdir(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/train/target')
        cond_train_name_list = os.listdir(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/train/condition')
        log.info(f'train data : {len(tar_train_name_list)}')
        tar_val_name_list = os.listdir(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/val/target')
        cond_val_name_list = os.listdir(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/val/condition')
        log.info(f'val data : {len(tar_val_name_list)}')
        
        tar_train_df_list = []
        for tar_train_name in tar_train_name_list:
            tar_train_df = pd.read_csv(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/train/target/{tar_train_name}')
            tar_train_df_list.append(tar_train_df)
        cond_train_df_list = []
        for cond_train_name in cond_train_name_list:
            cond_train_df = pd.read_csv(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/train/condition/{cond_train_name}')
            cond_train_df_list.append(cond_train_df)
        
        tar_val_df_list = []
        for tar_val_name in tar_val_name_list:
            tar_val_df = pd.read_csv(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/val/target/{tar_val_name}')
            tar_val_df_list.append(tar_val_df)
        cond_val_df_list = []
        for cond_val_name in cond_val_name_list:
            cond_val_df = pd.read_csv(f'/data/ts/{args.dataset_name}/{args.group}/{args.column_type}/{col}/val/condition/{cond_val_name}')
            cond_val_df_list.append(cond_val_df)
        
        # =========================================================================
        # dataset 생성
        print('dataset 생성')
        Train_Dataset = dam.AnomalyDataset(
            tar_df_list=tar_train_df_list,
            cond_df_list=cond_train_df_list
        )
        Val_Dataset = dam.AnomalyDataset(
            tar_df_list=tar_val_df_list,
            cond_df_list=cond_val_df_list
        )

        # =========================================================================
        # device & model 생성
        device = tum.get_device(args.device)
        log.info(f'device: {device}')

        args = parameter_setup(
            args=args,
            feature_length=len(tar_train_df_list[0].columns.values),
            condition_num=len(cond_train_df_list[0].columns.values)
        )
        model = Transformer.Model(args)

        if args.ddp == "use":
            rank, world_size = set_device()
            print(rank, world_size)
            Train_Sampler = DistributedSampler(
                Train_Dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=True
            )
            Val_Sampler = DistributedSampler(
                Val_Dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=True
            )
            model.cuda()
            model = DDP(model, device_ids=[device], output_device=device)
        else:
            world_size = args.num_workers
            Train_Sampler = SequentialSampler(Train_Dataset)
            Val_Sampler = SequentialSampler(Val_Dataset)
        model.to(device)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # =========================================================================
        # dataloader 생성            
        print('train dataloader 생성 중...')
        Train_Dataloader = DataLoader(
            Train_Dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=world_size, 
            pin_memory=True, 
            sampler=Train_Sampler
        )

        print('validation dataloader 생성 중...')
        Val_Dataloader = DataLoader(
            Val_Dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=world_size, 
            pin_memory=True, 
            sampler=Val_Sampler
        )
        # =========================================================================
        # optimizer
        optimizer = tum.get_optimizer(
            base='torch',
            method=args.optimizer_name,
            model=model,
            learning_rate=args.learning_rate
        )

        # loss function
        loss_function = tum.get_loss_function(
            base='torch',
            method='MSE')

        # scheduler
        scheduler_cosine = tum.get_scheduler(
            base='torch',
            method='CosineAnnealingLR',
            optimizer=optimizer,
            total_iter=args.total_iter,
            warmup_iter=args.warmup_iter
        )
        scheduler = LearningRateWarmUP(
            optimizer=optimizer,
            warmup_iteration=args.warmup_iter,
            target_lr=args.learning_rate,
            after_scheduler=scheduler_cosine
        )

        # =========================================================================
        torch.autograd.set_detect_anomaly(True)
        if args.retrain:
            model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_weight}'))
            print('\n>>> re-training')
        else:
            args.start_epoch = 0

        # =========================================================================
        # 학습 
        condition_order = utm.get_condition_order(
            args_setting=args_setting,
            save_path=root_save_path,
            except_arg_list=['epochs', 'device', 'column_type', 'c_out', 'enc_in', 'dec_in']
        )

        # 저장경로 생성
        model_save_path = f'{root_save_path}/{condition_order}/{args.column_type}/{col}'
        os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
        with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
            json.dump(args_setting, f, indent='\t', ensure_ascii=False)
        print(f'condition order: {condition_order}')

        # train
        save_dic = trm.train(
            model=model,
            purpose=args.purpose,
            start_epoch=args.start_epoch,
            epochs=args.epochs,
            train_dataloader=Train_Dataloader,
            validation_dataloader=Val_Dataloader,
            uni_class_list=None,
            device=device,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
            reset_class=None,
            model_save_path=model_save_path
        )

        # ==
        save_parameter = {}
        for k, v in vars(args).items():
            if k != "local_rank":
                save_parameter[k] = v
        save_dic['hyper_parameters'] = save_parameter
        save_dic['parameter_num'] = parameter_count
        best_epoch = save_dic['epoch'] + 1
        best_model_name = f'epoch{str(best_epoch).zfill(4)}'
        torch.save(save_dic, f'{model_save_path}/{best_model_name}/best.ckpt')
        # ==

        col_hh, col_mm, col_ss = utm.time_measure(col_start)
        log.info(f'{col} 학습 진행 시간: {col_hh}:{col_mm}:{col_ss}\n')
    whole_hh, whole_mm, whole_ss = utm.time_measure(whole_start)
    log.info(f'전체 학습 진행 시간: {whole_hh}:{whole_mm}:{whole_ss}\n')

        # # 변수 초기화
        # best_loss = float('inf')
        # history = {'best':{'epoch':0, 'loss':0}} 

        # # 학습 진행
        # start = time.time()
        # for epoch in range(start_epoch, epochs):
        #     history[f'epoch {epoch+1}'] = {'train_loss':0, 'val_loss':0}
        #     print(f'======== {epoch+1:2d}/{epochs} ========')
        #     print("lr: ", optimizer.param_groups[0]['lr'])

        #     # train
        #     model.train()
        #     _, _, _, _, train_loss = tm.get_output(
        #         mode='train',
        #         purpose=purpose,
        #         dataloader=Train_Dataloader,
        #         model=model,
        #         device=device,
        #         loss_function=loss_function,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         amp=amp,
        #         max_grad_norm=max_grad_norm
        #     )
        #     print(f"epoch {epoch+1} train loss {train_loss:.6f}")

        #     # validation
        #     model.eval()
        #     with torch.no_grad():
        #         val_pred_label_ary, _, _, val_true_label_ary, val_loss = get_output(
        #             mode='val',
        #             purpose=purpose,
        #             model=model, 
        #             dataloader=Val_Dataloader,
        #             device=device,
        #             loss_function=loss_function,
        #         )
        #     scheduler.step()

        #     # 학습 이력 저장
        #     history[f'epoch {epoch+1}']['train_loss'] = train_loss
        #     history[f'epoch {epoch+1}']['val_loss'] = val_loss
            

        #      # 최적의 모델 변수 저장        
        #     if val_loss < best_loss:
        #         best_loss = val_loss
        #         best_epoch = epoch + 1
        #         history['best']['epoch'] = best_epoch
        #         history['best']['loss'] = best_loss
        #         best_model_wts = copy.deepcopy(model.state_dict())

        #         # best result save
        #         best_model_name = f'epoch{str(best_epoch).zfill(4)}'
        #         os.makedirs(f'{model_save_path}/{best_model_name}', exist_ok=True)
        #         torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')

        #         # 이전의 최적 모델 삭제
        #         for i in range(epoch, 0, -1):
        #             pre_best_model_name = f'epoch{str(i).zfill(4)}'
        #             try:
        #                 shutil.rmtree(f'{model_save_path}/{pre_best_model_name}')
        #                 print(f'이전 모델 {pre_best_model_name} 삭제')
        #             except FileNotFoundError:
        #                 pass
            
        #     # history 저장
        #     with open(f'{model_save_path}/train_history.json', 'w', encoding='utf-8') as f:
        #         json.dump(history, f, indent='\t', ensure_ascii=False)
    
        #     print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')

        #     h, m, s = utils.time_measure(start)
        #     print(f'걸린시간: {h}시간 {m}분 {s}초\n')

        # # last result save
        # os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
        # torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')

        # # 최적의 학습모델 불러오기
        # print(f'best: {best_epoch}')
        # model.load_state_dict(best_model_wts)











            # # 기록
            # result_save_path = f'{root_save_path}/{condition_order}/{args.column_type}/{train_col}'
            # os.makedirs(result_save_path, exist_ok=True)
            # with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
            #     json.dump(vars(args), f, indent='\t', ensure_ascii=False)

            # save_parameter = {}
            # for k, v in vars(args).items():
            #     if k != "local_rank":
            #         save_parameter[k] = v
            # if args.ddp == "use":
            #     state_dict_temp = model.module.state_dict()
            # else:
            #     state_dict_temp = model.state_dict()
            # if val_loss <= best_loss:
            #     history[0]['best_epoch'] = epoch
            #     save_dic = {
            #         "epoch" : epoch,
            #         "train_loss" : train_loss,
            #         "val_loss" : val_loss,
            #         "state_dict" : state_dict_temp,
            #         "hyper_parameters" : save_parameter,
            #         "parameter_num" : parameter_count
            #     }
            #     print(f"{args.group} {args.column_type} {train_col} best model save")
            #     torch.save(save_dic, f'{result_save_path}/best.ckpt')
            #     best_loss = val_loss
            #     print(f"epoch : {epoch}, train_loss : {train_loss}, val_loss : {val_loss}")
            # else:
            #     print(f"epoch : {epoch}, train_loss : {train_loss}, val_loss : {val_loss}")
            #     print(f"{args.group} {args.column_type} {train_col} non_save")
            # history.append({"epoch" : str(epoch).zfill(4) , "train loss" : train_loss, "val loss" : val_loss})
            # with open(f'{result_save_path}/history.json', "w") as f:
            #     json.dump(history, f, indent="\t")
