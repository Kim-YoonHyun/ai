import sys
import os
sys.path.append(os.getcwd())
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
from mylocalmodules import tokenizer as tkm

from mylocalmodules import trainutils as tum
# from mylocalmodules import classificationutils as clm


def model_test(test_string_split_list, 
               test_id_split_list, 
               b_e_list, 
               test_idx_list,
               root_path, 
               weight_path, 
               device_num, 
               batch_size, 
               max_seq_len,
               use_custom, 
               custom_token_dict_name, 
               label2id_dict, 
               id2label_dict
    ):
    # ================================================================================
    # 데이터로더 생성
    print('make validation dataloader...')
    
    # ===================================================================    
    # 모델 생성
    print('get device')
    
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
                whole_pred_label_list[i].append(pred_label)
                
                true_label = id2label_dict[str(true_id)]
                whole_true_label_list[i].append(true_label)

    # confusion matrix 생성
    # whole_ss_list = []
    # whole_tl_list = []
    # whole_pl_list = []
    # for ss_list, tl_list, pl_list in zip(test_string_split_list, \
    #                                      whole_true_label_list, whole_pred_label_list):
    #     for ss, tl, pl in zip(ss_list, tl_list, pl_list):
    #         whole_ss_list.append(ss)
    #         whole_tl_list.append(tl)
    #         whole_pl_list.append(pl)
    # confusion_matrix = clm.make_confusion_matrix(
    #     mode='label2id', 
    #     true_list=whole_tl_list, 
    #     pred_list=whole_pl_list, 
    #     label2id_dict=label2id_dict
    # )
    
    # ===================================================================  
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
        
        # 문장이 나뉜 경우
        if b_e in ['b', 'e']:
            
            temp_text = whole_data['text']
            
            # 첫 문장인 경우
            if len(front_text) == 0:
                cor = 0
            # 첫 문장이 아닌 경우
            else:
                cor = 1
                temp_text = f' {temp_text}' # 앞에 띄어쓰기 추가
                    
            temp_span_list = whole_data['spans']
            for temp_span in temp_span_list:
                start = temp_span['start'] + len(front_text) + cor
                end = temp_span['end'] + len(front_text) + cor
                label = temp_span['label']
                
                new_span = {"start":start, "end":end, "label":label}
                each_data['spans'].append(new_span)
            front_text += temp_text
            
            # 마지막 부분인 경우            
            if b_e == 'e':
                # 지금까지 쌓아온 문장 저장
                each_data['text'] = front_text
                # 인덱스 저장
                each_data['index'] = whole_data['index']
                # 합쳐진 데이터 저장                
                new_whole_data_list.append(each_data)
                
                # 초기화
                front_text = ''
                each_data = {"text": None, "spans": [], 'index':None}
                
        # 문장이 나뉘지 않은 경우
        else:
            new_whole_data_list.append(whole_data)
    
    # # 번호 예측 진행
    # new_whole_data_list = list(map(ncm.num_classification, new_whole_data_list))
    return new_whole_data_list