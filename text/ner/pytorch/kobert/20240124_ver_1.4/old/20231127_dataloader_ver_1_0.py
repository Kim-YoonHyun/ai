import sys
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler

with open('./kobert_token2idx_dict.json', 'r', encoding='utf-8-sig') as f:
    token2idx_dict = json.load(f)
key_list = list(token2idx_dict.keys())
key_list.sort()
# def get_example_dict(data_path, uni_label_list):
#     # read data
#     with open(data_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     # example dict
#     # example_dict_list = []
#     # for data in data_line_list:
#     #     string, label = data.split('\t')
        
#     #     # string
#     #     string_split_list = string.split()
        
#     #     # label idx
#     #     label_split_list = label.split()
#     #     label_idx_list = []
#     #     for label_split in label_split_list:
#     #         if label_split in uni_label_list:
#     #             label_idx = uni_label_list.index(label_split)
#     #         else:
#     #             label_idx = uni_label_list.index('UNK')
#     #         label_idx_list.append(label_idx)

#     #     example_dict = {
#     #         'string_split_list':string_split_list,
#     #         'label_idx_list':label_idx_list
#     #     }
#     #     example_dict_list.append(example_dict)
#     return example_dict_list


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
        # print(string_split_list)
        # print(id_split_list)
        # print()
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
        # print(whole_token_list)
        # print(label_ids_list)
        # print()
        
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
    # # ==
    # input_ids_list = input_ids_list[:150]
    # attention_mask_list = attention_mask_list[:150]
    # token_type_ids_list = token_type_ids_list[:150]
    # label_ids_list = label_ids_list[:150]
    # # ==
    dataset = TensorDataset(input_ids_list, attention_mask_list, token_type_ids_list, label_ids_list)
    return dataset


def get_dataloader(feature_list, batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True):
    
    # dataset
    dataset = get_dataset(feature_list)
    
    # dataloader
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler, 
    )
    return dataloader
    
    

