'''
<<< 2023.08.10 >>>
아래의 순서로 세팅 진행 필요

pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
pip install -r requirements.txt
export TORCH_CUDA_ARCH_LIST=8.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip uninstall numpy
pip install 'numpy<1.22'
'''
import sys
import os
import argparse
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
from transformers import AdamW # 인공지능 모델의 초기값 지정 함수를 아담으로 지정한다.
from transformers.optimization import get_cosine_schedule_with_warmup

# local modules
from models import network as net
from mylocalmodules import dataloader as dam

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import utils as utm


def main():
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--phase')
    parser.add_argument('--dataset_name')
    parser.add_argument('--purpose', type=str)

    # train variable
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--loss_function_name', type=str, default='CrossEntropyLoss')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='optimizer learning rate')
    parser.add_argument('--gamma', type=float, default=0.95)
    
    # dataloader variable    
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)
    
    # model parameters
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--pad', type=bool, default=True)
    parser.add_argument('--pair', type=bool, default=False)

    args = parser.parse_args()
    args_setting = vars(args)
    utm.envs_setting(42)
    
    
    root_path = args.root_path
    phase = args.phase
    dataset_name = args.dataset_name
    purpose = args.purpose
    
    device_num = args.device_num
    epochs = args.epochs
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm 
    dropout_p = args.dropout_p
    loss_function_name = args.loss_function_name
    learning_rate = args.learning_rate
    gamma = args.gamma
    
    shuffle = args.shuffle
    drop_last = args.drop_last
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    
    max_len=args.max_len
    warmup_ratio = args.warmup_ratio
    pad = args.pad
    pair = args.pair
    
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'train.log',
        time_handler=True
    )
    whole_start = time.time()
    # =========================================================================
    # dataset 불러오기
    dataset_path = f'{root_path}/datasets'
    with open(f'{dataset_path}/{dataset_name}/train_data.json', 'r', encoding='utf-8-sig') as f:
        train_data = json.load(f)
    train_string_list = train_data['string']
    train_label_list = train_data['label']
    uni_label_list = np.unique(train_label_list).tolist()
    with open(f'{dataset_path}/{dataset_name}/val_data.json', 'r', encoding='utf-8-sig') as f:
        val_data = json.load(f)
    val_string_list = val_data['string']
    val_label_list = val_data['label']
    
    # =========================================================================
    print('get dataloader')
    raw_kobert_model, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    Train_Dataloader = dam.get_dataloader(
        string_list=train_string_list,
        label_list=train_label_list,
        bert_tokenizer=bert_tokenizer, 
        max_len=max_len, 
        pad=pad, 
        pair=pair,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    Val_Dataloader = dam.get_dataloader(
        string_list=val_string_list,
        label_list=val_label_list,
        bert_tokenizer=bert_tokenizer, 
        max_len=max_len, 
        pad=pad, 
        pair=pair,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # ===================================================================    
    print('get device')
    device = tum.get_device(device_num)
    
    print('get_model')
    model = net.Kobert(
        bert=raw_kobert_model,
        hidden_size=768,
        num_classes=len(uni_label_list),
        dropout_p=dropout_p,
    )
    model.to(device)
    
    # ===================================================================    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    loss_function = tum.get_loss_function(
        base='torch',
        method=loss_function_name
    )
    t_total = len(Train_Dataloader) * epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    # ============================================================
    # save setting
    save_path = f'{root_path}/trained_model/{phase}/{dataset_name}'
    condition_order = utm.get_condition_order(
        args_setting=args_setting,
        save_path=save_path,
        except_arg_list=['epochs']
    )
    model_save_path = f'{save_path}/{condition_order}'
    os.makedirs(f'{save_path}/{condition_order}', exist_ok=True)
    
    # class dict 불러오기
    with open(f'{save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    
    with open(f'{dataset_path}/{dataset_name}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)
    class_dict = dataset_info['class']['dict']
    uni_class_list = list(class_dict.keys())
    
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
        uni_class_list=uni_class_list, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        reset_class=None, 
        model_save_path=model_save_path
    )

        
if __name__  == '__main__':
    main()