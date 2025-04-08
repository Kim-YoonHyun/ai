'''
<<< 2023.08.10 >>>
python = 3.9.x
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
pip install -r requirements.txt
'''
import time
import sys
import os
import json
import argparse

from mylocalmodules import tokenizer as tkm
from mylocalmodules import dataloader as dam
from mylocalmodules import lossfunction as lfm

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import trainutils as tum
from sharemodule import train as trm
from sharemodule import utils as utm

import torch
from transformers import AdamW #, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForTokenClassification
from tokenization_kobert import KoBertTokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", default="./model", type=str, help="Path to save, load model")
    parser.add_argument('--phase')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--custom_tokenizer')
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument("--epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    args = parser.parse_args()
    args_setting = vars(args)
    
    # =========================================================================
    # varivables
    root_path = args.root_path
    phase = args.phase    
    dataset_name = args.dataset_name
    custom_token_dict_name = args.custom_tokenizer
    device_num = args.device_num
    epochs = args.epochs
    batch_size = args.batch_size
    
    max_seq_len = args.max_seq_len
    random_seed = args.random_seed
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    gradient_accumulation_steps = args.gradient_accumulation_steps
    adam_epsilon = args.adam_epsilon
    max_grad_norm = args.max_grad_norm
    max_steps = args.max_steps
    warmup_steps = args.warmup_steps
    
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'train.log',
        time_handler=True
    )
    whole_start = time.time()
    
    # ================================================================================
    # tokenizer
    # /home/kimyh/anaconda3/envs/kca_module/lib/python3.9/site-packages/transformers/tokenization_utils_base.py
    use_custom = os.path.isfile(f'{root_path}/token_dict/{custom_token_dict_name}')
    
    # ================================================================================
    # 데이터 불러오기
    print('데이터 불러오기')
    dataset_path = f'{root_path}/datasets/{dataset_name}'
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
    if use_custom:
        with open(f'{root_path}/token_dict/{custom_token_dict_name}', 'r', encoding='utf-8-sig') as f:
            tokenizer_dict = json.load(f)
    else:
        model_path = '/home/kimyh/python/ai/text/ner/pytorch/kobert/pretrained'
        kobert_tokenizer = KoBertTokenizer.from_pretrained(model_path)
    
    # ================================================================================
    # 데이터로더 생성    
    print('make train dataloader ...')
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    if use_custom:
        print('add_underbar')
        # train_string_split_list = list(map(tkm.add_underbar, train_string_split_list))
        print('custom tokenize')
        train_feature_list = dam.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=train_string_split_list,
            whole_id_split_list=train_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=max_seq_len
        )
    else:
        train_feature_list = dam.get_feature(
            whole_string_split_list=train_string_split_list,
            whole_id_split_list=train_id_split_list,
            tokenizer=kobert_tokenizer,
            max_seq_len=max_seq_len,
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    train_dataloader = dam.get_dataloader(
        feature_list=train_feature_list, 
        batch_size=batch_size
    )
    
    # val dataloader
    print('make validation dataloader ...')
    if use_custom:
        val_string_split_list = list(map(tkm.add_underbar, val_string_split_list))
        val_feature_list = dam.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=val_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=max_seq_len
        )
    else:
        val_feature_list = dam.get_feature(
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=val_id_split_list,
            tokenizer=kobert_tokenizer,
            max_seq_len=max_seq_len,
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    val_dataloader = dam.get_dataloader(
        feature_list=val_feature_list, 
        batch_size=batch_size
    )
    
    # ===================================================================
    # device
    print('get device')
    device = tum.get_device(device_num)
    
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
    if max_steps > 0:
        t_total = max_steps
        epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate, 
        eps=adam_epsilon
    )
    loss_function = lfm.LossFunction(gradient_accumulation_steps)
    scheduler = tum.get_scheduler(
        base='transformers',
        method='linear_schedule_with_warmup',
        optimizer=optimizer,
        warmup_iter=warmup_steps,
        total_iter=t_total
    )
    
    # ============================================================
    # save setting
    root_save_path = f'{root_path}/trained_model/{phase}/{dataset_name}'
    args_setting['label2id_dict'] = label2id_dict
    args_setting['id2label_dict'] = id2label_dict
    condition_order = utm.get_condition_order(
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
    epochs = int(epochs)
    _ = trm.train(
        model=model, 
        purpose=None, 
        start_epoch=0, 
        epochs=epochs, 
        train_dataloader=train_dataloader, 
        validation_dataloader=val_dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        reset_class=None, 
        model_save_path=model_save_path
    )
    print(f'condition order: {condition_order}')
    


if __name__ == '__main__':
    main()

