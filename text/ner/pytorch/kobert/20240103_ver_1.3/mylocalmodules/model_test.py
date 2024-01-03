import sys
import os
from tqdm import tqdm
import numpy as np
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import BertConfig, BertForTokenClassification
from tokenization_kobert import KoBertTokenizer

# local modules
from mylocalmodules import dataloader as dam
from mylocalmodules import num_classification as ncm
from mylocalmodules import tokenizer as tkm

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum
from sharemodule import classificationutils as clm


def model_test(test_string_split_list, test_id_split_list, 
               b_e_list, test_idx_list, 
               root_path, weight_path, 
               device_num, batch_size, max_seq_len,
               use_custom, custom_token_dict_name,
               label2id_dict, id2label_dict):
    
    # ================================================================================
    # tokenizer
    print('tokenizer 생성')
    if use_custom:
        with open(f'{root_path}/token_dict/{custom_token_dict_name}', 'r', encoding='utf-8-sig') as f:
            tokenizer_dict = json.load(f)
    else:
        model_path = '/home/kimyh/python/ai/text/ner/pytorch/kobert/pretrained'
        tokenizer = KoBertTokenizer.from_pretrained(model_path)
    
    # ================================================================================
    # 데이터로더 생성
    print('make validation dataloader ...')
    
    # 데이터 로더 생성
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    val_string_split_list = copy.deepcopy(test_string_split_list)
    if use_custom:
        val_string_split_list = list(map(tkm.add_underbar, val_string_split_list))
        val_feature_list = dam.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=test_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=max_seq_len 
        )
    else:
        val_feature_list = dam.get_feature(
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=test_id_split_list,
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
    # 모델 생성
    print('get device')
    device = tum.get_device(device_num)
    
    print('get model config')
    if use_custom:
        model_config = BertConfig(
            vocab_size=len(tokenizer_dict),
            pad_token_id=1,
            id2label=id2label_dict,
            label2id=label2id_dict
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
    print('get_model')
    if use_custom:
        model = BertForTokenClassification(config=model_config)
    else:
        model = BertForTokenClassification.from_pretrained(model_path, config=model_config)
    model.to(device)
    
    # 학습된 가중치 로딩
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    # ===================================================================    
    print('예측 진행')
    pred_list = []
    true_label_ids_list = []
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
            
            # pred
            pred = pred.to('cpu').detach().numpy().tolist() # 메모리 해제
            pred_list.extend(pred)
            
            true_label_ids = inputs["labels"].detach().cpu().tolist()
            true_label_ids_list.extend(true_label_ids)
            
    pred_ary = np.array(pred_list)
    pred_ary = np.argmax(pred_ary, axis=2)
    true_label_ids_ary = np.array(true_label_ids_list)
    
    # ===================================================================    
    # true, pred 라벨 정리
    whole_pred_label_list = []
    whole_true_label_list = []
    for _ in range(len(true_label_ids_list)):
        whole_pred_label_list.append([])
        whole_true_label_list.append([])
        
    for i in range(true_label_ids_ary.shape[0]):
        for j in range(true_label_ids_ary.shape[1]):
            if true_label_ids_ary[i, j] != pad_token_label_id:
                pred_id = pred_ary[i][j]
                true_id = true_label_ids_ary[i][j]
                
                pred_label = id2label_dict[str(pred_id)]
                pred_label = pred_label.replace('B_', '')
                pred_label = pred_label.replace('I_', '')
                if pred_label in ['시도', '시군구', '도로명', '읍면동', '건물번호_상세', '기타주소']:
                    pred_label = '주소'
                whole_pred_label_list[i].append(pred_label)
                
                true_label = id2label_dict[str(true_id)]
                true_label = true_label.replace('B_', '')
                true_label = true_label.replace('I_', '')
                if true_label in ['시도', '시군구', '도로명', '읍면동', '건물번호_상세', '기타주소']:
                    true_label = '주소'
                whole_true_label_list[i].append(true_label)

    # confusion matrix 생성
    print('confusion matrix 생성')
    whole_ss_list = []
    whole_tl_list = []
    whole_pl_list = []
    
    con_label2id_dict = {}
    n = 0
    for key in label2id_dict.keys():
        key = key.replace('B_', '')
        if 'I_' in key:
            key = key.replace('I_', '')
            n -= 1
        if key in ['시도', '시군구', '도로명', '읍면동', '건물번호_상세', '기타주소']:
            key = '주소'
            n = 5
        con_label2id_dict[key] = n
        n += 1
        
    for ss_list, tl_list, pl_list in zip(test_string_split_list, \
                                         whole_true_label_list, whole_pred_label_list):
        for ss, tl, pl in zip(ss_list, tl_list, pl_list):
            whole_ss_list.append(ss)
            whole_tl_list.append(tl)
            whole_pl_list.append(pl)
    confusion_matrix = clm.make_confusion_matrix(
        mode='label2id', 
        true_list=whole_tl_list, 
        pred_list=whole_pl_list, 
        label2id_dict=con_label2id_dict
    )
    
    # ===================================================================    
    # 결과 데이터 생성
    print('결과 데이터 생성중..')
    whole_data_list = []
    for ss_list, pl_list, fi in zip(test_string_split_list, whole_pred_label_list, test_idx_list):
        string = ' '.join(ss_list)
        each_data = {"text": string, "spans": None, 'index':fi}
        
        len_list = list(map(len, ss_list))
        start = 0
        span_list = []
        
        for ll, pl in zip(len_list, pl_list):
            if pl != 'O':
                end = start + ll
                span = {"start":start, "end":end, "label":pl}
                span_list.append(span)
            start = start + ll + 1
        each_data['spans'] = span_list
        whole_data_list.append(each_data)

    # ===================================================================    
    # 나눠진 데이터 합치기
    print('나눠진 데이터 재결합')
    new_whole_data_list = []
    front_text = ''
    each_data = {"text": None, "spans": [], 'index':None}
    for whole_data, b_e in zip(whole_data_list, b_e_list):
        if b_e in ['b', 'e']:
            temp_text = whole_data['text']
            
            temp_span_list = whole_data['spans']
            for temp_span in temp_span_list:
                start = temp_span['start'] + len(front_text)
                end = temp_span['end'] + len(front_text)
                label = temp_span['label']
                
                new_span = {"start":start, "end":end, "label":label}
                each_data['spans'].append(new_span)
            front_text += temp_text
            if b_e == 'e':
                each_data['text'] = front_text
                each_data['index'] = whole_data['index']
                
                new_whole_data_list.append(each_data)
                front_text = ''
                each_data = {"text": None, "spans": [], 'index':None}
        else:
            new_whole_data_list.append(whole_data)
    
    # 번호 예측 진행
    new_whole_data_list = list(map(ncm.num_classification, new_whole_data_list))    
    
    return new_whole_data_list, confusion_matrix