'''
<<< 2023.08.10 >>>
python = 3.9.x
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
pip install -r requirements.txt

pip install pandas==1.3.5
pip install konlpy
pip install psutil
pip3 install torch torchvision torchaudio
'''
import time
import sys
import os
import json
import argparse

from mylocalmodules import tokenizer as tkn
from mylocalmodules import dataloader as dl
from mylocalmodules import lossfunction as lf

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lu
from sharemodule import trainutils as tu
from sharemodule import train as tr
from sharemodule import utils as u

import torch
from transformers import AdamW #, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForTokenClassification
from tokenization_kobert import KoBertTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, help="Path to save, load model")
    parser.add_argument('--root_dataset_path')
    parser.add_argument('--phase')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--custom_tokenizer')
    parser.add_argument('--purpose')
    
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument("--epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--loss_function_name", type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    args = parser.parse_args()
    return args


def trainer(ars):
    # args_setting = vars(args)
    
    # =========================================================================
    # varivables
    # root_path = args.root_path
    # phase = args.phase    
    # dataset_name = args.dataset_name
    # custom_token_dict_name = args.custom_tokenizer
    # device_num = args.device_num
    # epochs = args.epochs
    # batch_size = args.batch_size
    
    # max_seq_len = args.max_seq_len
    # random_seed = args.random_seed
    # learning_rate = args.learning_rate
    # weight_decay = args.weight_decay
    # gradient_accumulation_steps = args.gradient_accumulation_steps
    # adam_epsilon = args.adam_epsilon
    # max_grad_norm = args.max_grad_norm
    # max_steps = args.max_steps
    # warmup_steps = args.warmup_steps
    
    # u.envs_setting(ars['random_seed'])
    
    # =========================================================================
    # # log 생성
    # log = lo.get_logger(
    #     get='TRAIN',
    #     root_path=args.root_path,
    #     log_file_name=f'train.log',
    #     time_handler=True
    # )
    # whole_start = time.time()
    
    # ================================================================================
    # tokenizer
    # /home/kimyh/anaconda3/envs/kca_module/lib/python3.9/site-packages/transformers/tokenization_utils_base.py
    
    
    # ================================================================================
    # 데이터 불러오기
    print('데이터 불러오기')
    dataset_path = f'{ars["root_dataset_path"]}/datasets/{ars["dataset_name"]}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    label2id_dict = json_dataset['dict']['label2id']
    id2label_dict = json_dataset['dict']['id2label']
    
    train_string_split_list = json_dataset['data']['train']['string']
    train_id_split_list = json_dataset['data']['train']['id']
    val_string_split_list = json_dataset['data']['validation']['string']
    val_id_split_list = json_dataset['data']['validation']['id']
    
    # ================================================================================
    # tokenizer
    print('tokenizer 생성')
    custom_path = f'{ars["root_path"]}/token_dict/{ars["custom_tokenizer"]}'
    use_custom = os.path.isfile(custom_path)
    if use_custom:
        with open(custom_path, 'r', encoding='utf-8-sig') as f:
            tokenizer_dict = json.load(f)
    else:
        model_path = '/home/kimyh/python/ai/text/ner/pytorch/kobert/pretrained_tokenizer'
        kobert_tokenizer = KoBertTokenizer.from_pretrained(model_path)
    
    # ================================================================================
    # 데이터로더 생성    
    print('make train dataloader ...')
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    if use_custom:
        print('add_underbar')
        # train_string_split_list = list(map(tkm.add_underbar, train_string_split_list))
        print('custom tokenize')
        train_feature_list = dl.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=train_string_split_list,
            whole_id_split_list=train_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=ars['max_seq_len']
        )
    else:
        train_feature_list = dl.get_feature(
            whole_string_split_list=train_string_split_list,
            whole_id_split_list=train_id_split_list,
            tokenizer=kobert_tokenizer,
            max_seq_len=ars['max_seq_len'],
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    train_dataloader = dl.get_dataloader(
        feature_list=train_feature_list, 
        batch_size=ars['batch_size']
    )
    
    # val dataloader
    print('make validation dataloader ...')
    if use_custom:
        val_string_split_list = list(map(tkn.add_underbar, val_string_split_list))
        val_feature_list = dl.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=val_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=ars['max_seq_len']
        )
    else:
        val_feature_list = dl.get_feature(
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=val_id_split_list,
            tokenizer=kobert_tokenizer,
            max_seq_len=ars['max_seq_len'],
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    val_dataloader = dl.get_dataloader(
        feature_list=val_feature_list, 
        batch_size=ars['batch_size']
    )
    
    # ===================================================================
    # device
    print('get device')
    device = tu.get_device(ars['device_num'])
    
    # model config
    # /home/kimyh/anaonda3/envs/kca_module/lib/python3.9/site-packages/transformers/configuration_utils.py
    print('get model config')
    if use_custom:
        model_config = BertConfig(
            vocab_size=len(tokenizer_dict),  # 사용하는 어휘 사전 크기
            pad_token_id=1,
            id2label=id2label_dict,
            label2id=label2id_dict
            # hidden_size=768,
        )
    else:
        num_labels = len(label2id_dict)
        model_config = BertConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            finetuning_task='naver-ner',
            id2label=id2label_dict,
            label2id=label2id_dict
        )
    print(f'vocab size: {model_config.vocab_size}')
    
    # model
    print('get model')
    if use_custom:
        model = BertForTokenClassification(config=model_config)
    else:
        model = BertForTokenClassification.from_pretrained(model_path, config=model_config)
    model.to(device)
    
    # =========================================================
    # optimizer & schedule & loss function
    # if ars['max_steps'] > 0:
    #     t_total = ars['max_steps']
    #     epochs = ars['max_steps'] // (len(train_dataloader) // ars['gradient_accumulation_steps']) + 1
    # else:
    t_total = len(train_dataloader) // ars['gradient_accumulation_steps'] * ars['epochs']
        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': ars['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=ars['learning_rate'], 
        eps=ars['adam_epsilon']
    )
    # loss_function = lf.LossFunction(gradient_accumulation_steps)
    criterion = lf.LossFunction(
        base='torch',
        method=ars['loss_function_name'],
        pad_idx=pad_token_label_id
    )
        
    scheduler = tu.get_scheduler(
        base='transformers',
        method='linear_schedule_with_warmup',
        optimizer=optimizer,
        warmup_iter=ars['warmup_steps'],
        total_iter=t_total
    )
    
    # ============================================================
    # save setting
    root_save_path = f'{ars["root_path"]}/trained_model/{ars["phase"]}/{ars["dataset_name"]}'
    ars['label2id_dict'] = label2id_dict
    ars['id2label_dict'] = id2label_dict
    condition_order = tu.get_condition_order(
        args_setting=ars,
        save_path=root_save_path,
        except_arg_list=['epochs', 'batch_size', 'purpose']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(ars, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = tr.train(
        model=model, 
        purpose=ars['purpose'], 
        start_epoch=0, 
        epochs=ars['epochs'], 
        train_dataloader=train_dataloader, 
        validation_dataloader=val_dataloader, 
        device=device, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=ars['max_grad_norm'], 
        model_save_path=model_save_path,
        id2label_dict=id2label_dict,
        ignore_idx=pad_token_label_id            
    )
    print(f'condition order: {condition_order}')
    

def main():
    args_ = get_args()
    args_setting = vars(args_)
    u.envs_setting(args_setting['random_seed'])
    
    # =========================================================================
    # log 생성
    log = lu.get_logger(
        get='TRAIN',
        root_path=args_setting['root_path'],
        log_file_name='train.log',
        time_handler=True
    )
    trainer(args_setting)
    

if __name__ == '__main__':
    main()
    

