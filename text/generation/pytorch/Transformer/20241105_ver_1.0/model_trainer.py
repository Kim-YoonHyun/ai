'''
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install transformers
conda install -c conda-forge statsmodels
conda install -c conda-forge gcc=12.1.0
pip install pykrx
pip install scikit-learn
'''
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



from mylocalmodules import dataloader as dam
from model import network as net

sys.path.append('/home/kimyh/python/ai')
from sharemodule import lossfunction as lfm
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm


import torch
from torch.utils.data import DataLoader, Dataset, random_split
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import MarianMTModel, MarianTokenizer
# from transformers import AutoTokenizer

from tqdm import tqdm
# from konlpy.tag import Okt



def trainer():
    root_path = '/home/kimyh/python/project/transformer'
    device_num = 1
    
    # train parameter
    epochs = 100
    train_p = 0.9
    max_grad_norm = 1
    
    dropout_p = 0.1
    loss_function_name = 'CrossEntropyLoss'
    optimizer_name = 'Adam'
    learning_rate = 0.0005
    eta_min = 0.00001
    scheduler_name = 'CosineAnnealingLR'
    random_seed = 42
    
    # network parameter for train
    # batch_size = 64
    # max_len = 128
    # d_model = 256
    # d_ff = 512
    # n_heads = 8
    # enc_layer_num = 6
    # dec_layer_num = 6
     
    # network parameter for test
    batch_size = 1
    max_len = 12
    d_model = 8
    d_ff = 32
    n_heads = 4
    enc_layer_num = 4
    dec_layer_num = 4
    
    # seed 지정
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # 데이터 불러오기
    data_df = pd.read_csv('/home/kimyh/python/ai/text/generation/pytorch/Transformer/data/chatdata.csv', encoding='utf-8-sig')
    # data_df = pd.read_csv('/home/kimyh/python/ai/text/generation/pytorch/Transformer/data/1_구어체(1).csv', encoding='utf-8-sig')
    # data_df = data_df.iloc[:100000, :]
    kor_list = data_df['원문'].tolist()
    eng_list = data_df['번역문'].tolist()
    eng_list = ['</s> ' + eng for eng in eng_list]
    
    # =========================================================================
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    
    # model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    
    print("eos_idx = ", eos_idx)
    print("pad_idx = ", pad_idx)
    print(f'vocab size = {vocab_size}')
    
    # =========================================================================    
    # 한국어 token --> id
    kor_ids_pad_list = tokenizer(
        kor_list, 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
    ).input_ids
    # add_special_tokens = True (default)면 마지막 토큰에 <eos> 가 붙어 나옴
    # truncation = True: max_len 보다 길면 끊고 <eos> 집어넣어버림
    # src에 <eos>가 있는 게 반드시 좋은 건지는 알 수 없지만 그냥 붙여봤어요..
    
    # =========================================================================    
    # 영어 token --> id
    eng_ids_pad_list = tokenizer(
        eng_list, 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
    ).input_ids
    
    # =========================================================================
    # dataset
    Text_Dataset = dam.TextDataset(
        x_list=kor_ids_pad_list,
        y_list=eng_ids_pad_list
    )
    train_num = int(len(Text_Dataset) * train_p)
    val_num = len(Text_Dataset) - train_num
    Train_Dataset, Val_Dataset = random_split(Text_Dataset, [train_num, val_num])
    # =========================================================================
    # dataloader
    train_dataloader = dam.get_dataloader(Train_Dataset, batch_size)
    val_dataloader = dam.get_dataloader(Val_Dataset, batch_size)
    # =========================================================================
    # device 
    # tum.see_device()
    device = tum.get_device(device_num)
    
    # src = tokenizer(temp_x_list, padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids # pt: pytorch tensor로 변환
    # src = src.to(device)
    # temp_y_list = ['</s> ' + s for s in temp_y_list]
    # trg = tokenizer(temp_y_list, padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids

    # =========================================================================
    # model        
    model = net.Transformer(
        device=device,
        max_len=max_len,
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        pad_idx=pad_idx,
        enc_layer_num=enc_layer_num,
        dec_layer_num=dec_layer_num,
        dropout_p=dropout_p
    )
    model.to(device)
    
    # see parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"총 파라미터 수: {total_params}")
    # sys.exit()
    

    # pre-trained
    start_epoch = 0
    # torch.autograd.set_detect_anomaly(True)
    
    # =========================================================================
    # optimizer
    optimizer = tum.get_optimizer(
        base='torch',
        method=optimizer_name,
        model=model,
        learning_rate=learning_rate
    )

    # loss function
    criterion = lfm.LossFunction(
        base='torch',
        method=loss_function_name,
        pad_idx=pad_idx
    )
    
    # scheduler
    total_iter = epochs * (1+len(train_dataloader) // batch_size)
    warmup_iter = int(total_iter * 0.1)
    scheduler = tum.get_scheduler(
        base='torch',
        method=scheduler_name,
        eta_min=eta_min,
        optimizer=optimizer,
        total_iter=total_iter,
        warmup_iter=warmup_iter
    )
    
    # =========================================================================
    # save setting
    model_save_path = f"{root_path}/trained_model/test"
    # condition_order = tum.get_condition_order(
    #     args_setting=new_args_setting,
    #     save_path=root_save_path,
    #     except_arg_list=['epochs', 'batch_size']
    # )
    # model_save_path = f'{root_save_path}/{condition_order}'
    # os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    # with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        # json.dump(new_args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    # print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose='None', 
        start_epoch=start_epoch, 
        epochs=epochs, 
        train_dataloader=train_dataloader, 
        validation_dataloader=val_dataloader, 
        device=device, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        model_save_path=model_save_path
    )


def main():
    # args = get_args()
    # args_setting = vars(args)
    # args_setting = temporal_type_setting(args, args_setting)
    # args_setting = activation_setting(args, args_setting)

    # =========================================================================
    

    # =========================================================================
    # # log 생성
    # log = lom.get_logger(
    #     get='TRAIN',
    #     root_path=args_setting['root_path'],
    #     log_file_name='train.log',
    #     time_handler=True
    # )
    # =========================================================================
    # 학습 진행
    trainer()

if __name__ == "__main__":
    main()    
