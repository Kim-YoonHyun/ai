from datetime import date
import sys
import copy
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
from mylocalmodules import tokenizer as tkm

sys.path.append('/home/kimyh/python')
from preprocessmodule.text.mylocalmodules import preprocessutils as ppm

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import classificationutils as clm
from sharemodule import utils as utm


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

def del_underbar(split):
    if '_' in split:
        split = split.replace('_', '')
    return split

def main():
    
    
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')
    parser.add_argument('--stan_num', type=int)
    parser.add_argument('--dummy_label', type=bool)
    parser.add_argument('--test_data', type=str)

    args = parser.parse_args()
    args_setting = vars(args)
    
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    stan_num = args.stan_num
    dummy_label = args.dummy_label
    test_data = args.test_data
    
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
        
    device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    use_pretrained = args_setting['use_pretrained']
    token_dict_name = args_setting['token_dict_name']
    random_seed = args_setting['random_seed']
    max_seq_len = args_setting['max_seq_len']
    task = args_setting['task']
    dataset_name = args_setting['dataset_name']
    utm.envs_setting(random_seed)
    

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    
    # ================================================================================
    # 데이터 불러오기
    # dataloader
    print('데이터 불러오기')
    dataset_path = f'{root_path}/datasets/{dataset_name}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    label2id_dict = json_dataset['dict']['label2id']
    id2label_dict = json_dataset['dict']['id2label']
    
    if not test_data:
        # 학습시 활용한 데이터셋의 validation 데이터 활용
        test_string_split_list = json_dataset['data']['validation']['string']
        print(111111111111111)
        sys.exit()
    else:
        # 직접 지정한 데이터 활용
        try:
            with open(f'./test_data/{test_data}', 'r', encoding='utf-8-sig') as f:
                test_data_list = json.load(f)
                test_data_list = [test_data_list]
        except json.decoder.JSONDecodeError:
            test_data_list = []
            with open(f'./test_data/{test_data}', 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.replace('\n', '')
                    line.strip()
                    if line[-1] == '}':
                        json_line = json.loads(line)
                        test_data_list.append(json_line)

        test_idx_list = []
        for test_data in test_data_list:
            test_idx_list.append(test_data['index'])
        
        # 전처리        
        print('데이터 전처리')
        remove_re_df = pd.read_csv('/home/kimyh/python/project/kca/KoBERT-NER/whole_module/supports/remove_re_df.csv', encoding='utf-8-sig')
        remove_string_df = pd.read_csv('/home/kimyh/python/project/kca/KoBERT-NER/whole_module/supports/remove_string_df.csv', encoding='utf-8-sig')
        test_string_list = []
        for test_data in tqdm(test_data_list):
            text = test_data['text']
            new_text = ppm.preprocess(
                string_list=[text],
                remove_enter=True,
                remove_emoji=True,
                adjust_blank=True,
                remove_re=True,
                remove_string=True,
                remove_string_df=remove_string_df,
                remove_re_df=remove_re_df
            )[0]
            test_string_list.append(new_text)
        
        
        test_string_split_list = []
        for test_string in test_string_list:
            # test_string_split = test_string.split(' ')
            # print(test_string_split)
            # print(len(test_string_split))
            temp_result = tkm.okt_base_tokenizer(test_string)
            test_string_split = temp_result[0]
            # number_position = temp_result[1]
            # space_position = temp_result[2]
            # sentence = temp_result[3]
            test_string_split = tkm.temp_space(test_string_split)
            test_string_split_list.append(test_string_split)
            
        # 너무 긴 문장 제어
        print('긴 문장 제어...')
        result_list = dam.separate_long_text(
            string_split_list=test_string_split_list, 
            idx_list=test_idx_list,
            stan_num=stan_num
        )
        test_string_split_list = result_list[0]
        b_e_list = result_list[1]
        new_test_idx_list = result_list[2]
    
    # ================================================================================
    # tokenizer
    if use_pretrained:
        model_path = 'monologg/kobert'
        tokenizer = KoBertTokenizer.from_pretrained(model_path)
    else:
        with open(f'./token_dict/{token_dict_name}', 'r', encoding='utf-8-sig') as f:
            tokenizer_dict = json.load(f)
    
    # ================================================================================
    num_labels = len(label2id_dict)
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    
    # val dataloader
    if dummy_label:
        whole_id_split_list = []
        for string_split_list in test_string_split_list:
            id_split_list = len(string_split_list) * [0]
            whole_id_split_list.append(id_split_list)
    else:
        whole_id_split_list = json_dataset['data']['validation']['id']
    
    # 데이터로더 생성
    print('make validation dataloader ...')
    val_string_split_list = copy.deepcopy(test_string_split_list)
    if use_pretrained:
        val_feature_list = dam.get_feature(
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=whole_id_split_list,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    else:
        val_feature_list = dam.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=val_string_split_list,
            whole_id_split_list=whole_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=max_seq_len 
        )
    val_dataloader = dam.get_dataloader(
        feature_list=val_feature_list, 
        batch_size=batch_size
    )
    # ===================================================================    
    device = tum.get_device(device_num)
    
    print('get device')
    if use_pretrained:
        config = BertConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            finetuning_task=task,
            id2label=id2label_dict,
            label2id=label2id_dict
        )
    else:
        config = BertConfig(
            vocab_size=len(tokenizer_dict),
            pad_token_id=1,
            id2label=id2label_dict,
            label2id=label2id_dict
        )
    print(config)
        
    # model
    print('get_model')
    if use_pretrained:
        model = BertForTokenClassification.from_pretrained(model_path, config=config)
    else:
        model = BertForTokenClassification(config=config)
    model.to(device)
    
    weight = torch.load(f'{trained_model_path}/{trained_model}/weight.pt')
    model.load_state_dict(weight)
    model.to(device)
    
    # ===================================================================    
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
    
    # true, pred 라벨 정리
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

    # confusion matrix 생성
    whole_ss_list = []
    whole_tl_list = []
    whole_pl_list = []
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
        label2id_dict=label2id_dict
    )
    
    print('결과 데이터 생성중..')
    whole_data_list = []
    for ss_list, pl_list, fi in zip(test_string_split_list, whole_pred_label_list, new_test_idx_list):
        ss_list = list(map(del_underbar, ss_list))
        string = ' '.join(ss_list)
        each_data = {"text": string, "spans": None, 'index':fi}
        
        len_list = list(map(len, ss_list))
        start = 0
        span_list = []
        
        for ss, ll, pl in zip(ss_list, len_list, pl_list):
            if pl != 'O':
                end = start + ll
                span = {"start":start, "end":end, "label":pl}
                span_list.append(span)
            start = start + ll + 1
        each_data['spans'] = span_list
        whole_data_list.append(each_data)

    # 나눠진 데이터 합치기
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

    # 정상적으로 잘 합쳐졌는지 테스트    
    for new_whole_data in new_whole_data_list:
        text = new_whole_data['text']
        span_list = new_whole_data['spans']        
        for span in span_list:
            start = span['start']
            end = span['end']
            label = span['label']
            print(text[start:end], start, end, label)
    

    print('데이터 저장중..')
    today = date.today()
    today_string = today.strftime('%Y%m%d')
    print(today_string)
    print(len(new_whole_data_list))
    with open(f"./test_result/pred_{today_string}_{len(new_whole_data_list)}_data_pretrained_{use_pretrained}.jsonl", "w", encoding='utf-8') as jsonl_file:
        for entry in new_whole_data_list:
            json_string = json.dumps(entry, ensure_ascii=False)
            jsonl_file.write(json_string+"\n")
            
    # {"text": "", "spans": [{"start": 102, "end": 106, "label": "상품"}, {"start": 130, "end": 134, "label": "상품"}], "index": 49}
    
        
if __name__  == '__main__':
    main()