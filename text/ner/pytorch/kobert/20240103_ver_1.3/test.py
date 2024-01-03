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

import torch
from transformers import BertConfig, BertForTokenClassification
from tokenization_kobert import KoBertTokenizer

# local modules
from mylocalmodules import dataloader as dam
from mylocalmodules import tokenizer as tkm
from mylocalmodules import model_test as mtm
from mylocalmodules import num_classification as ncm

sys.path.append('/home/kimyh/python')
from preprocessmodule.text.mylocalmodules import preprocessutils as ppm

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import classificationutils as clm
from sharemodule import utils as utm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')
    parser.add_argument('--device_num')
    parser.add_argument('--stan_num', type=int)
    parser.add_argument('--dummy_label', type=bool)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--result_save_path', type=str)

    args = parser.parse_args()
    args_setting = vars(args)
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    device_num = args.device_num
    stan_num = args.stan_num
    dummy_label = args.dummy_label
    test_data = args.test_data
    result_save_path = args.result_save_path
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TEST',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    
    # ==========================================================
    # 모델에 적용된 변수 불러오기
    print('변수 불러오기')
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
    
    try:
        device_num = int(device_num)
    except ValueError:
        device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    custom_token_dict_name = args_setting['custom_tokenizer']
    random_seed = args_setting['random_seed']
    max_seq_len = args_setting['max_seq_len']
    label2id_dict = args_setting['label2id_dict']
    id2label_dict = args_setting['id2label_dict']
    dataset_name = args_setting['dataset_name']
    
    utm.envs_setting(random_seed)
    weight_path = f'{trained_model_path}/{trained_model}/weight.pt'
    use_custom = os.path.isfile(f'{root_path}/token_dict/{custom_token_dict_name}')
    
    # ================================================================================
    # 데이터 불러오기
    print('데이터 불러오기')
    dataset_path = f'{root_path}/datasets/{dataset_name}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    test_string_split_list = json_dataset['data']['validation']['string']
    whole_id_split_list = json_dataset['data']['validation']['id']
    
    new_test_idx_list = [0] * len(test_string_split_list)
    b_e_list = ['be'] * len(test_string_split_list)
    
    if test_data:
        expention = test_data.split('.')[-1]
        
        if expention in ['csv', 'xlsx', 'xls', 'txt']:
            data_df = utm.read_df(f'{root_path}/test_data/{test_data}')
            string_list = data_df['string'].to_list()
            test_data_list = []
            for idx, string in enumerate(string_list):
                temp_dict = {'text':string, 'index':idx}
                test_data_list.append(temp_dict)
        elif expention == 'jsonl':
            test_data_list = utm.read_jsonl(f'{root_path}/test_data/{test_data}')
        
        test_idx_list = []
        for test_data in test_data_list:
            test_idx_list.append(test_data['index'])
        
        # ================================================================================
        # 전처리        
        print('텍스트 전처리')
        remove_re_df = pd.read_csv(f'{root_path}/supports/remove_re_df.csv', encoding='utf-8-sig')
        remove_string_df = pd.read_csv(f'{root_path}/supports/remove_string_df.csv', encoding='utf-8-sig')
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
    
        # ================================================================================
        print('token 화 진행')
        test_string_split_list = []
        for test_string in test_string_list:
            temp_result = tkm.okt_base_tokenizer(test_string)
            test_string_split = temp_result[0]
            # number_position = temp_result[1]
            # space_position = np.where(np.array(temp_result[2]) == 1, ' ', '').tolist()
            # sentence = temp_result[3]
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
        
        # 라벨 생성
        whole_id_split_list = []
        for string_split_list in test_string_split_list:
            id_split_list = len(string_split_list) * [0]
            whole_id_split_list.append(id_split_list)
    
    # 예측 진행
    whole_pred_data_list, confusion_matrix = mtm.model_test(
        test_string_split_list=test_string_split_list,
        test_id_split_list=whole_id_split_list,
        b_e_list=b_e_list,
        test_idx_list=new_test_idx_list,
        root_path=root_path,
        weight_path=weight_path,
        device_num=device_num,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        use_custom=use_custom,
        custom_token_dict_name=custom_token_dict_name,
        label2id_dict=label2id_dict,
        id2label_dict=id2label_dict
        
    )
    # ===================================================================    
    # 비식별화 진행
    text_split_list = []
    for pred_data in whole_pred_data_list:
        text = pred_data['text']
        new_text = ''
        span_list = pred_data['spans']        
        pre_end = 0
        
        # span (개체명 인식) 이 존재하는 경우
        if len(span_list) > 0:
            for span in span_list:
                start = span['start']
                end = span['end']
                label = span['label']
                
                new_text += text[pre_end:start]
                new_text += f'({label})'
                
                print(text[start:end], start, end, label)
    
                pre_end = end
            new_text += text[end:]
        else:
            new_text = text
        new_text = new_text.replace('_', '')
        new_text_split = new_text.split(' ')
        text_split_list.append(new_text_split)
        
    # ===================================================================    
    # 저장
    print('데이터 저장중..')
    today = date.today()
    today_string = today.strftime('%Y%m%d')
    print(today_string)
    print(len(whole_pred_data_list))
    os.makedirs(result_save_path, exist_ok=True)
    with open(f"{result_save_path}/pred_{today_string}_{len(whole_pred_data_list)}_data_csutom_{use_custom}.jsonl", "w", encoding='utf-8') as f:
        for entry in whole_pred_data_list:
            json_string = json.dumps(entry, ensure_ascii=False)
            f.write(json_string+"\n")
            
    confusion_matrix.to_csv(f'{result_save_path}/confusion_matrix.csv', encoding='utf-8-sig')
            
    # {"text": "", "spans": [{"start": 102, "end": 106, "label": "상품"}, {"start": 130, "end": 134, "label": "상품"}], "index": 49}
    
        
if __name__  == '__main__':
    main()