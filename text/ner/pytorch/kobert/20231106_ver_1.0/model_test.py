import sys
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 
import torch
from transformers import AdamW #, get_linear_schedule_with_warmup

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from tokenization_kobert import KoBertTokenizer

# local modules
# from models import network as net
from mylocalmodules import dataloader as dam
from mylocalmodules import utils as utm

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import classificationutils as clm


# def make_confusion_matrix(mode, true_list, pred_list, label2id_dict=None, id2label_dict=None):
#     if mode == 'label2id':
#         uni_label_list = label2id_dict.keys()
#     elif mode == 'id2label':
#         uni_label_list = id2label_dict.values()
#     # matrix
#     matrix = []
#     for i in range(len(uni_label_list)):
#         matrix.append([])
#         for _ in range(len(uni_label_list)):
#             matrix[i].append(0)
    
#     # count
#     if mode == 'label2id':
#         for t, p in zip(true_list, pred_list):
#             t_i = label2id_dict[t]
#             p_i = label2id_dict[p]
#             matrix[t_i][p_i] += 1
#     elif mode == 'id2label':
#         for t_i, p_i in zip(true_list, pred_list):
#             matrix[t_i][p_i] += 1
        
#     whole_sum = np.sum(matrix)
#     true_sum_list = np.sum(matrix, axis=-1)
#     pred_sum_list = np.sum(matrix, axis=-2)
    
#     # make matrix
#     correct_sum = 0
#     for i in range(len(matrix)):
#         correct_count = matrix[i][i]
#         correct_sum += correct_count
#         pred_sum = pred_sum_list[i]
#         true_sum = true_sum_list[i]
        
#         precision = correct_count / pred_sum
#         recall = correct_count / true_sum
#         f1_score = 2*precision*recall / (precision + recall)
        
#         matrix[i].extend([None, precision, recall, f1_score, true_sum])
#     whole_accuracy = correct_sum / whole_sum
    
#     # index & column
#     index_list = uni_label_list.copy()
#     column_list = uni_label_list.copy()
#     column_list.extend(['accuracy', 'precision', 'recall', 'f1 score', 'count'])
    
#     # confusion matrix
#     confusion_matrix = pd.DataFrame(matrix, index=index_list, columns=column_list)
#     confusion_matrix['accuracy'][0] = whole_accuracy
    
#     return confusion_matrix


def main():
    test_string_split_list = [
        ['이곳은', '아무것도', '없는', '3층 빌딩', '입니다.'],
        ['저는 회사에서 일하고 있는 대리', '홍길동',],
        # ['이벤트 개최지가 충북 도청 근처라고 하셨나요?']
    ]
        
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')

    args = parser.parse_args()
    args_setting = vars(args)
    
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
    device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    model_type = args_setting['model_type']
    random_seed = args_setting['random_seed']
    max_seq_len = args_setting['max_seq_len']
    task = args_setting['task']
    dataset_name = args_setting['dataset_name']
    
    utm.envs_setting(random_seed)

    # pretrained model path
    model_path_dict = {
        'kobert': 'monologg/kobert',
        'distilkobert': 'monologg/distilkobert',
        'bert': 'bert-base-multilingual-cased',
        'kobert-lm': 'monologg/kobert-lm',
        'koelectra-base': 'monologg/koelectra-base-discriminator',
        'koelectra-small': 'monologg/koelectra-small-discriminator',
    }
    # model_class
    model_class_dict = {
        'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
        'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
        'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
        'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
        'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
        'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    }
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    # ================================================================================
    # tokenizer
    model_path = model_path_dict[model_type]
    tokenizer = model_class_dict[model_type][2].from_pretrained(model_path)
    
    # ================================================================================
    # dataloader
    dataset_path = f'{root_path}/datasets/{dataset_name}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    label2id_dict = json_dataset['dict']['label2id']
    id2label_dict = json_dataset['dict']['id2label']
    

    
    num_labels = len(label2id_dict)
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    
    # val dataloader
    def string_split(string):
        result = string.split(' ')
        return result 
        
    # whole_string_split_list = list(map(string_split, test_string_list))
    whole_id_split_list = []
    for string_split_list in test_string_split_list:
        id_split_list = []
        for _ in string_split_list:
            id_split_list.append(0)
        whole_id_split_list.append(id_split_list)
    
    print('make validation dataloader ...')
    val_feature_list = dam.get_feature(
        whole_string_split_list=test_string_split_list,
        whole_id_split_list=whole_id_split_list,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_token_label_id=pad_token_label_id, #-100
        mask_padding_with_zero=True
    )
    val_dataloader = dam.get_dataloader(
        feature_list=val_feature_list, 
        batch_size=batch_size
    )
    # ===================================================================    
    print('get device')
    config_class, model_class, _ = model_class_dict[model_type]
    device = tum.get_device(device_num)

    config = config_class.from_pretrained(
        model_path,
        num_labels=num_labels,
        finetuning_task=task,
        id2label=id2label_dict,
        label2id=label2id_dict
    )
    
    print('get_model')
    # model
    model = model_class.from_pretrained(model_path, config=config)
    model.to(device)
    weight = torch.load(f'{trained_model_path}/{trained_model}/weight.pt')
    model.load_state_dict(weight)
    model.to(device)
    
    # ===================================================================    
    pred_list = []
    true_label_ids_list = []
    whole_string_split_list = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(val_dataloader):
            input_ids = batch_data[0]
            input_ids = input_ids.to(device)
            
            attention_mask = batch_data[1]
            attention_mask = attention_mask.to(device)
            
            token_type_ids = batch_data[2]
            token_type_ids = token_type_ids.to(device)
            
            labels = batch_data[3]
            labels = labels.to(device)
            
            # input 생성
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'token_type_ids':token_type_ids
            }
            outputs = model(**inputs)
            
            _, pred = outputs[:2]
            # b_label = labels
            
            # pred
            pred = pred.to('cpu').detach().numpy().tolist() # 메모리 해제
            pred_list.extend(pred)
            
            true_label_ids = inputs["labels"].detach().cpu().tolist()
            true_label_ids_list.extend(true_label_ids)
            
            # 문장
            input_ids_ary = input_ids.to('cpu').detach().numpy() # 메모리 해제
            for input_ids in input_ids_ary:
                token_list = tokenizer.convert_ids_to_tokens(input_ids)
                for unk_token in ['[CLS]', '[SEP]', '[PAD]']:
                    while True:
                        try:
                            token_list.remove(unk_token)
                        except ValueError:
                            break
                # ▁  
                # _ 하고는 다른 문자임
                string = ''.join(token_list)
                if string[0] == '▁':
                    string = string[1:]
                string_split_list = string.split('▁')
                whole_string_split_list.append(string_split_list)
                
    pred_ary = np.array(pred_list)
    pred_ary = np.argmax(pred_ary, axis=2)
    true_label_ids_ary = np.array(true_label_ids_list)
    
    # result
    whole_pred_label_list = []
    whole_true_label_list = []
    for _ in range(len(true_label_ids_list)):
        whole_pred_label_list.append([])
        whole_true_label_list.append([])
        
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    for i in range(true_label_ids_ary.shape[0]):
        for j in range(true_label_ids_ary.shape[1]):
            if true_label_ids_ary[i, j] != pad_token_label_id:
                pred_id = pred_ary[i][j]
                true_id = true_label_ids_ary[i][j]
                
                whole_pred_label_list[i].append(id2label_dict[str(pred_id)])
                whole_true_label_list[i].append(id2label_dict[str(true_id)])

    # save
    whole_ss_list = []
    whole_tl_list = []
    whole_pl_list = []
    for ss_list, tl_list, pl_list in zip(test_string_split_list, \
                                         whole_true_label_list, whole_pred_label_list):
        # input_string = ' '.join(ss_list)
        print(ss_list)
        print(pl_list)
        print()
        # for ss, tl, pl in zip(ss_list, tl_list, pl_list):
        #     whole_ss_list.append(ss)
        #     whole_tl_list.append(tl)
        #     whole_pl_list.append(pl)
    sys.exit()
    
    
    
    # confusion matrix
    confusion_matrix = clm.make_confusion_matrix(
        mode='label2id', 
        true_list=whole_tl_list, 
        pred_list=whole_pl_list, 
        label2id_dict=label2id_dict
    )
    # confusion_matrix.to_csv('./aaa.csv', encoding='utf-8-sig')
    # print(confusion_matrix)
    sys.exit()
        
    for label, id in label2id_dict.items():
        print(label, id)        
        sys.exit()
    print(label2id_dict)    
    
    
    # confusion_matrix = clm.make_confusion_matrix(
    #     uni_class_list=uni_label_list, 
    #     true=whole_ti_list, 
    #     pred=whole_pi_list
    # )
    # print(result_df)
    # print(confusion_matrix)
        
if __name__  == '__main__':
    main()