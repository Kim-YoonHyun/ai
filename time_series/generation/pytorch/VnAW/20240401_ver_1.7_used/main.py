'''
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install transformers
'''
import sys
import os
sys.path.append(os.getcwd())
import time
import json
import copy
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd

import torch
# from torch.utils.data import DataLoader, SequentialSampler


# from utils.warm_up import LearningRateWarmUP

from mylocalmodules import dataloader as dam
from model_origin_2 import network as net

sys.path.append('/home/kimyh/python/ai')
from sharemodule import lossfunction as lfm
from sharemodule import logutils as lom
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm


if __name__ == "__main__":
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
    parser.add_argument('--sampler_name', type=str, default='SequentialSampler')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--pre_trained', type=str, default='None')
    
    # network parameter
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=1064, help='dimension of fcn')
    parser.add_argument('--x_len', type=int, help='prediction sequence length')
    parser.add_argument('--y_len', type=int, help='prediction sequence length')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--embed_type', type=str)
    parser.add_argument('--temporal_type', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

    # layer num parameter
    parser.add_argument('--enc_layer_num', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--dec_layer_num', type=int, default=1, help='num of decoder layers')
    
    # activation parameter
    parser.add_argument('--enc_activation', type=str, default='relu', help='activation')
    parser.add_argument('--dec_activation', type=str, default='gelu', help='activation')
    

    
    args = parser.parse_args()
    args_setting = vars(args)
    
    # basic parameter
    root_path = args.root_path
    root_dataset_path = args.root_dataset_path
    phase = args.phase
    dataset_name = args.dataset_name
    reduce_num = args.reduce_num
    scale = args.scale
    purpose = args.purpose
    
    # train paremater
    device_num = args.device_num
    epochs = args.epochs
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm
    dropout_p = args.dropout_p
    loss_function_name = args.loss_function_name
    optimizer_name = args.optimizer_name
    learning_rate = args.learning_rate
    scheduler_name = args.scheduler_name
    random_seed = args.random_seed
    sampler_name = args.sampler_name
    
    shuffle = args.shuffle
    drop_last = args.drop_last
    num_workers = args.num_workers
    pin_memory = args.pin_memory

    pre_trained = args.pre_trained
    
    # network parameter
    d_model = args.d_model
    d_ff = args.d_ff
    x_len = args.x_len
    y_len = args.y_len
    n_heads = args.n_heads
    embed_type = args.embed_type
    if embed_type == 'pure':
        temporal_type = None
        args_setting['temporal_type'] = None
    else:
        temporal_type = args.temporal_type
    
    # layer num
    enc_layer_num = args.enc_layer_num
    dec_layer_num = args.dec_layer_num
    
    # activation
    if enc_layer_num == 0:
        enc_activation = None
        args_setting['enc_activation'] = None
    else:
        enc_activation = args.enc_activation
    dec_activation = args.dec_activation

    # =========================================================================
    utm.envs_setting(random_seed)

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name='train.log',
        time_handler=True
    )
    whole_start = time.time()
    
    # =========================================================================
    # 데이터 불러오기
    print('데이터 불러오기')
    dataset_path = f'{root_dataset_path}/datasets/{dataset_name}'
    
    # norm 불러오기
    norm_path = '/'.join(dataset_path.split('/')[:-2])
    with open(f'{norm_path}/norm.json', 'r', encoding='utf-8-sig') as f:
        norm = json.load(f)
    
    # 갯수 줄이기
    train_num = len(os.listdir(f'{dataset_path}/train/x'))
    val_num = len(os.listdir(f'{dataset_path}/val/x'))
    if reduce_num == 0:
        train_size = train_num
        val_size = val_num
    else:
        train_size = reduce_num
        val_size = int(reduce_num * 0.01)
    
    print('갯수 줄이기')
    train_idx_ary = np.random.choice(range(train_num), size=train_size, replace=False)
    val_idx_ary = np.random.choice(range(val_num), size=val_size, replace=False)
    
    # 데이터 리스트 생성
    name_dict = {}
    
    # train, val 별로 진행
    print('train, val 데이터 리스트 불러오기')
    for tv in ['train', 'val']:
        name_dict[tv] = {}
        
        # x, y, mark 별로 진행
        for xym in ['x', 'y', 'mark']:
            name_list = os.listdir(f'{dataset_path}/{tv}/{xym}')
            name_list.sort()
            
            # 갯수 필터링            
            if tv == 'train':
                idx_ary = train_idx_ary
            if tv == 'val':
                idx_ary = val_idx_ary
            filter_name_list = np.array(name_list)[idx_ary].tolist()
            name_dict[tv][xym] = filter_name_list
    
    # =========================================================================
    # dataloader 생성
    print('train dataloader 생성 중...')
    Train_Dataloader = dam.get_dataloader(
        dataset_path=f'{dataset_path}/train', 
        x_name_list=name_dict['train']['x'], 
        y_name_list=name_dict['train']['y'], 
        mark_name_list=name_dict['train']['mark'], 
        scale=scale,
        norm=norm,
        n_heads=n_heads,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler_name=sampler_name
    )
    print('validation dataloader 생성 중...')
    Val_Dataloader = dam.get_dataloader(
        dataset_path=f'{dataset_path}/val', 
        x_name_list=name_dict['val']['x'], 
        y_name_list=name_dict['val']['y'], 
        mark_name_list=name_dict['val']['mark'], 
        scale=scale,
        norm=norm,
        n_heads=n_heads,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler_name=sampler_name
    )
    
    # =========================================================================
    # 출력 length 세팅
    temp_dataloader = copy.deepcopy(Train_Dataloader)
    x, mark, _, y, _, _ = next(iter(temp_dataloader))
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    mark = mark.cpu().detach().numpy()
    
    enc_feature_len = x.shape[-1]
    dec_feature_len = y.shape[-1]
    enc_temporal_len = mark.shape[-1]
    dec_temporal_len = mark.shape[-1]
    out_feature_len = dec_feature_len
    
    args_setting['enc_feature_len'] = enc_feature_len
    args_setting['dec_feature_len'] = dec_feature_len
    if embed_type == 'pure':
        args_setting['enc_temporal_len'] = None
        args_setting['dec_temporal_len'] = None
    else:
        args_setting['enc_temporal_len'] = enc_temporal_len
        args_setting['dec_temporal_len'] = dec_temporal_len
    args_setting['out_feature_len'] = out_feature_len
    
    # =========================================================================
    # device & model 생성
    tum.see_device()
    device = tum.get_device(device_num)
    
    # model        
    model = net.VnAW(
        x_len=x_len, 
        y_len=y_len,
        embed_type=embed_type, 
        d_model=d_model, 
        d_ff=d_ff, 
        n_heads=n_heads, 
        temporal_type=temporal_type, 
        enc_layer_num=enc_layer_num,
        dec_layer_num=dec_layer_num,
        enc_feature_len=enc_feature_len, 
        dec_feature_len=dec_feature_len,
        enc_temporal_len=enc_temporal_len,
        dec_temporal_len=dec_temporal_len,
        enc_act=enc_activation,
        dec_act=dec_activation,
        dropout_p=dropout_p,
        output_length=out_feature_len
    )
    model.to(device)
    
    # =========================================================================
    # optimizer
    optimizer = tum.get_optimizer(
        base='torch',
        method=optimizer_name,
        model=model,
        learning_rate=learning_rate
    )

    # loss function
    loss_function = lfm.LossFunction(
        base='torch',
        method=loss_function_name
    )
    # scheduler
    total_iter = epochs * (1 + len(name_dict['train']['x']) // batch_size)
    warmup_iter = int(total_iter * 0.1)
    scheduler = tum.get_scheduler(
        base='torch',
        method=scheduler_name,
        optimizer=optimizer,
        total_iter=total_iter,
        warmup_iter=warmup_iter
    )
    
    torch.autograd.set_detect_anomaly(True)
    if pre_trained != 'None':
        model.load_state_dict(torch.load(f'{pre_trained}'))
        print('\n>>> use pre-trained model')
    else:
        start_epoch = 0
    
    root_save_path = f'{root_path}/trained_model/{phase}/{dataset_name}'
    condition_order = tum.get_condition_order(
        args_setting=args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose=purpose, 
        start_epoch=0, 
        epochs=epochs, 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        reset_class=None, 
        model_save_path=model_save_path
    )
    print(condition_order)
    sys.exit()
    
    for col in col_list:
        # if ddp == "use":
        #     rank, world_size = set_device()
        #     print(rank, world_size)
        #     Train_Sampler = DistributedSampler(
        #         Train_Dataset, 
        #         num_replicas=world_size, 
        #         rank=rank, 
        #         shuffle=True
        #     )
        #     Val_Sampler = DistributedSampler(
        #         Val_Dataset, 
        #         num_replicas=world_size, 
        #         rank=rank, 
        #         shuffle=True
        #     )
        #     model.cuda()
        #     model = DDP(model, device_ids=[device], output_device=device)
        # else:
        #     world_size = num_workers
        #     Train_Sampler = SequentialSampler(Train_Dataset)
        #     Val_Sampler = SequentialSampler(Val_Dataset)

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
            # result_save_path = f'{root_save_path}/{condition_order}/{column_type}/{train_col}'
            # os.makedirs(result_save_path, exist_ok=True)
            # with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
            #     json.dump(vars(args), f, indent='\t', ensure_ascii=False)

            # save_parameter = {}
            # for k, v in vars(args).items():
            #     if k != "local_rank":
            #         save_parameter[k] = v
            # if ddp == "use":
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
            #     print(f"{group} {column_type} {train_col} best model save")
            #     torch.save(save_dic, f'{result_save_path}/best.ckpt')
            #     best_loss = val_loss
            #     print(f"epoch : {epoch}, train_loss : {train_loss}, val_loss : {val_loss}")
            # else:
            #     print(f"epoch : {epoch}, train_loss : {train_loss}, val_loss : {val_loss}")
            #     print(f"{group} {column_type} {train_col} non_save")
            # history.append({"epoch" : str(epoch).zfill(4) , "train loss" : train_loss, "val loss" : val_loss})
            # with open(f'{result_save_path}/history.json', "w") as f:
            #     json.dump(history, f, indent="\t")
