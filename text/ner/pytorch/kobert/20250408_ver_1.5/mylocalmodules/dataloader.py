import sys
import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler


def get_feature(whole_string_split_list, whole_id_split_list, tokenizer, max_seq_len,
                pad_token_label_id=-100, cls_token_segment_id=0,
                pad_token_segment_id=0, sequence_a_segment_id=0,
                mask_padding_with_zero=True):
    
    
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    # feature 생성
    feature_list = []
    for string_split_list, id_split_list in zip(whole_string_split_list, whole_id_split_list):
        whole_token_list = []
        label_ids_list = []

        for string_split, id_split in zip(string_split_list, id_split_list):
            token_list = tokenizer.tokenize(string_split)

            if not token_list:
                token_list = [unk_token]
            whole_token_list.extend(token_list)
            
            label_ids = [int(id_split)] + [pad_token_label_id]*(len(token_list) - 1)
            label_ids_list.extend(label_ids)
            
        # 너무 긴 경우 슬라이싱
        special_tokens_count = 2
        stan_num = max_seq_len - special_tokens_count
        if len(whole_token_list) > stan_num:
            whole_token_list = whole_token_list[:stan_num]
            label_ids_list = label_ids_list[:stan_num]
        
        # Add [CLS], [SEP] token
        whole_token_list = [cls_token] + whole_token_list + [sep_token]
        label_ids_list = [pad_token_label_id] + label_ids_list + [pad_token_label_id]
        token_type_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(whole_token_list)-1)
        
        # input ids
        input_ids = tokenizer.convert_tokens_to_ids(whole_token_list)

        # attention mask
        if mask_padding_with_zero:
            attention_mask = [1] * len(input_ids)
        else:
            attention_mask = [0] * len(input_ids)

        # padding
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + [pad_token_segment_id]*padding_length
        label_ids_list = label_ids_list + [pad_token_label_id]*padding_length

        # sys.exit()
        # feature list
        feature_list.append([input_ids, attention_mask, token_type_ids, label_ids_list])

    return feature_list


def get_dataset(feature_list):
    # Convert to Tensors and build dataset
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    label_ids_list = []
    for feature in feature_list:
        input_ids_list.append(feature[0])
        attention_mask_list.append(feature[1])
        token_type_ids_list.append(feature[2])
        label_ids_list.append(feature[3])
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long)
    token_type_ids_list = torch.tensor(token_type_ids_list, dtype=torch.long)
    label_ids_list = torch.tensor(label_ids_list, dtype=torch.long)
    
    # TensorDataset 은 Dataset 을 상속한 클래스로 
    # 학습데이터 X, 레이블 Y 만을 입력받는 단순한 구조의 Dataset 이다.
    # 오직 tensor 만을 입력받을 수 있으며 직접 작성은 불가능하다.
    dataset = TensorDataset(input_ids_list, attention_mask_list, token_type_ids_list, label_ids_list)
    return dataset


def get_dataloader(feature_list, 
                   batch_size, 
                   shuffle=False, 
                   drop_last=False, 
                   num_workers=1, 
                   pin_memory=True,
                   sampler_name='SequentialSampler'
                   ):
    
    # dataset
    dataset = get_dataset(feature_list)
    
    # dataloader
    if sampler_name == 'SequentialSampler':
        sampler = SequentialSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler, 
    )
    return dataloader
    

def custom_tokenize(
    unk_token, pad_token, cls_token, sep_token, mask_token,
    whole_string_split_list, 
    whole_id_split_list, 
    tokenizer_dict, 
    max_seq_len):

    # feature 생성
    feature_list = []
    for string_split_list, id_split_list in zip(whole_string_split_list, whole_id_split_list):
        
        # 특수 토큰 입력        
        string_split_list.insert(0, cls_token)
        string_split_list.append(sep_token)
        id_split_list.insert(0, -100)
        id_split_list.append(-100)
        
        # padding        
        pad_num = max_seq_len - len(string_split_list)
        if pad_num > 0:
            pad_token_list = pad_num * [pad_token]
            pad_label_list = pad_num * [-100]
            
            string_split_list.extend(pad_token_list)
            id_split_list.extend(pad_label_list)
        else:
            string_split_list = string_split_list[:max_seq_len - 1]
            string_split_list.append(sep_token)
            
            id_split_list = id_split_list[:max_seq_len - 1]
            id_split_list.append(-100)
        
        # input ids
        input_ids = []
        for string_split in string_split_list:
            try:
                token_idx = tokenizer_dict[string_split]
            except KeyError:
                token_idx = tokenizer_dict[unk_token]
            input_ids.append(token_idx)
        input_ids = np.array(input_ids)
        
        # attention mask
        attention_mask = np.where(input_ids != tokenizer_dict[pad_token], 1, 0)
        
        # token type ids
        token_type_ids = [0] * len(input_ids)
        
        # label ids list
        label_ids_list = id_split_list.copy()
            
        # 결과 리스트 생성
        feature_list.append([input_ids, attention_mask, token_type_ids, label_ids_list])

    return feature_list



# 너무 긴 문장 제어   
def separate_long_text(string_split_list, stan_num, idx_list=None): 
    b_e_list = []
    new_test_idx_list = []
    test_string_split_list = []
    
    if idx_list == None:
        idx_list = [0] * len(string_split_list)
        
    for test_string_split, test_idx in zip(tqdm(string_split_list), idx_list):
        # test_string_split = test_string.split(' ')
        if len(test_string_split) > stan_num:
            pre_start = 0
            pre_end = stan_num
            while True:
                temp_test_ss = test_string_split[pre_start:pre_end]
                if len(temp_test_ss) == 0:
                    # 끝
                    b_e_list[-1] = 'e'
                    break
                
                test_string_split_list.append(temp_test_ss)
                pre_start = pre_end
                pre_end += stan_num
                
                if len(temp_test_ss) < stan_num:
                    # 끝
                    b_e_list.append('e')
                    new_test_idx_list.append(test_idx)
                    break
                else:
                    new_test_idx_list.append(test_idx)
                    b_e_list.append('b')
        else:
            new_test_idx_list.append(test_idx)
            b_e_list.append('be')
            test_string_split_list.append(test_string_split)
    result_list = [test_string_split_list, b_e_list, new_test_idx_list]
    return result_list