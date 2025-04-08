import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import argparse

import torch
from transformers import BertConfig, BertForTokenClassification
from tokenization_kobert import KoBertTokenizer

# local modules
from mylocalmodules import dataloader as dam
from mylocalmodules import tokenizer as tkm
# from mylocalmodules import model_test as mtm
from mylocalmodules import symbolic_model as smm

sys.path.append('/home/kimyh/python')
from preprocessmodule.text.mylocalmodules import preprocessutils as ppm

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
# from sharemodule import classificationutils as clm
from sharemodule import utils as utm

def start_end_adjust(text, new_start, new_end):
    while True:
        label_text = text[new_start:new_end]
        
        # 앞 부분 공백 제거
        if label_text[0] == ' ':
            s_flag = 1
            new_start += 1
        else:
            s_flag = 0
            
        # 뒷부분 공백 제거
        if label_text[-1] == ' ':
            new_end -= 1
            e_flag = 1
        else:
            e_flag = 0
        
        if s_flag == 0 and e_flag == 0:
            break
    return new_start, new_end


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
    parser.add_argument('--la_sep', type=str)
    parser.add_argument('--see_detail', type=bool)

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
    la_sep = args.la_sep
    see_detail = args.see_detail
    
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
    
    use_custom = os.path.isfile(f'{root_path}/token_dict/{custom_token_dict_name}')
    
    # unique label_list
    symbolic_label_list = [
        "계좌번호",
        "사업자등록번호",
        "주민등록번호",
        "전화번호"
    ]
    
    uni_label_list = copy.deepcopy(symbolic_label_list)
    for key in label2id_dict.keys():
        key = key.replace('B▲', '')
        if 'I▲' in key:
            key = key.replace('I▲', '')
        uni_label_list.append(key)
    uni_label_list = list(set(uni_label_list))
    uni_label_list.sort()
    con_label2id_dict = {}
    con_id2label_dict = {}
    for n, uni_label in enumerate(uni_label_list):
        con_label2id_dict[uni_label] = n
        con_id2label_dict[n] = uni_label
            
    # ================================================================================
    # 데이터 불러오기
    print('데이터 불러오기')
    dataset_path = f'{root_path}/datasets/{dataset_name}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    whole_string_split_list = json_dataset['data']['validation']['string']
    whole_id_split_list = json_dataset['data']['validation']['id']
    new_test_idx_list = [0] * len(whole_string_split_list)
    b_e_list = ['be'] * len(whole_string_split_list)
    
    
    if test_data:
        # =========================================================================
        # 데이터 불러오기        
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
        
        # =========================================================================
        
        # support
        support_path = f'{root_path}/supports'
        
        # 전처리용 보조파일
        try:
            junk_df = pd.read_csv(f'{support_path}/junk_df.csv', encoding='utf-8-sig')
        except pd.errors.EmptyDataError:
            junk_df = pd.DataFrame()
        try:
            remove_string_df = pd.read_csv(f'{support_path}/remove_string_df.csv', encoding='utf-8-sig')
        except pd.errors.EmptyDataError:
            remove_string_df = pd.DataFrame()
        try:
            remove_re_df = pd.read_csv(f'{support_path}/remove_re_df.csv', encoding='utf-8-sig')
        except pd.errors.EmptyDataError:
            remove_re_df = pd.DataFrame()
        try:
            replace_df = pd.read_csv(f'{support_path}/replace_dict.csv', encoding='utf-8-sig')
        except pd.errors.EmptyDataError:
            replace_df = pd.DataFrame()
            
        # symbolic AI 용 보조파일
        with open(f'{support_path}/num_dict.json', encoding='utf-8-sig') as f:
            num_dict = json.load(f)
        
        # ================================================================================
        # 전처리        
        print('텍스트 전처리')
        string_list = []
        for test_data in test_data_list:
            text = test_data['text']
            string_list.append(text) 
        
        whole_string_list = ppm.preprocess(
            string_list=string_list,
            remove_junk=False,
            remove_re=True,
            remove_string=True,
            replace_string=False,
            remove_emoji=True,
            remove_enter=True,
            remove_kor=False,
            remove_eng=False,
            remove_num=False,
            adjust_blank=True,
            strip_string=True,
            junk_df=junk_df,
            remove_string_df=remove_string_df,
            remove_re_df=remove_re_df,
            replace_df=replace_df
        )
    
        # ================================================================================
        print('token 화 진행')
        whole_string_split_list = []
        number_position_list = []
        # space_position_list = []
        origin_space_list = []
        custom = tkm.CustomRule(root_path)
        
        # test_string_split_list = []
        for string in whole_string_list:
            temp_result = tkm.okt_base_tokenizer(string, custom)
            string_split_list = temp_result[0]
            number_position = temp_result[1]
            sentence = temp_result[3]
            origin_space = tkm.restoraion(string_split_list, number_position, sentence)
            
            whole_string_split_list.append(string_split_list)
            number_position_list.append(number_position)
            origin_space_list.append(origin_space)
            
        # 너무 긴 문장 제어
        result_list = dam.separate_long_text(
            string_split_list=whole_string_split_list, 
            idx_list=test_idx_list,
            stan_num=stan_num
        )
        whole_string_split_list = result_list[0]
        b_e_list = result_list[1]
        new_test_idx_list = result_list[2]
        
        # 더미 라벨 생성
        whole_id_split_list = []
        for string_split_list in whole_string_split_list:
            id_split_list = len(string_split_list) * [0]
            whole_id_split_list.append(id_split_list)
    
    # ================================================================================
    # tokenizer
    print('tokenizer 생성')
    if use_custom:
        with open(f'{root_path}/token_dict/{custom_token_dict_name}', 'r', encoding='utf-8-sig') as f:
            tokenizer_dict = json.load(f)
    else:
        model_path = f'{root_path}/data10/transformers'
        kobert_tokenizer = KoBertTokenizer.from_pretrained(model_path)
        
    # ================================================================================
    # val dataloader
    print('make test dataloader ...')
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    if use_custom:
        whole_string_split_list = list(map(tkm.add_underbar, whole_string_split_list))
        test_feature_list = dam.custom_tokenize(
            unk_token="[UNK]", 
            pad_token="[PAD]", 
            cls_token="[CLS]", 
            sep_token="[SEP]", 
            mask_token="[MASK]",
            whole_string_split_list=whole_string_split_list,
            whole_id_split_list=whole_id_split_list,
            tokenizer_dict=tokenizer_dict,
            max_seq_len=max_seq_len
        )
    else:
        test_feature_list = dam.get_feature(
            whole_string_split_list=whole_string_split_list,
            whole_id_split_list=whole_id_split_list,
            tokenizer=kobert_tokenizer,
            max_seq_len=max_seq_len,
            pad_token_label_id=pad_token_label_id, #-100
            mask_padding_with_zero=True
        )
    Test_Dataloader = dam.get_dataloader(
        feature_list=test_feature_list, 
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
    
    # 학습된 가중치 로딩    
    weight_path = f'{trained_model_path}/{trained_model}/weight.pt'
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    # ===================================================================
    logit_ary, true_label_ids_ary, _, _ = trm.model_test(
        model=model, 
        test_dataloader=Test_Dataloader, 
        device=device
    )
    
    pred_ary = np.argmax(logit_ary, axis=2)
    # ===================================================================
    # true, pred 라벨 정리
    whole_pred_label_list = []
    whole_true_label_list = []
    for _ in range(len(true_label_ids_ary)):
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
    
    print('결과 데이터 생성중..')
    whole_data_list = []
    for ss_list, pl_list, fi in zip(whole_string_split_list, whole_pred_label_list, new_test_idx_list):
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
    
    # =========================================================================
    whole_pred_data_list = new_whole_data_list
    num_list_dict = {'except':list(map(int, num_dict['except'].keys()))}
    for symbolic_label in symbolic_label_list:
        num_list_dict[symbolic_label] = list(map(int, num_dict[symbolic_label].keys()))

    whole_pred_data_list = smm.symbolic_ai(
        data_list=whole_pred_data_list, 
        num_list_dict=num_list_dict
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
                
                for bi in ['B▲', 'I▲']:
                    if bi in label:
                        label = label.replace(bi, '')
                
                new_text += text[pre_end:start]
                if see_detail:
                    new_text += f'{la_sep}{text[start:end]}({label}){la_sep}'
                else:
                    new_text += f'{la_sep}({label}){la_sep}'
                
                print(text[start:end], start, end, label)
                
                pre_end = end
            new_text += text[end:]
        else:
            new_text = text
        new_text = new_text.replace('_', '')
        new_text_split = new_text.split(' ')
        text_split_list.append(new_text_split)
        
    # ===================================================================    
    print('예측결과 NER 데이터 생성중..')
    # BIO 라벨 span 을 하나로 합침
    new_whole_data_list = []
    for iii, pred_data in enumerate(whole_pred_data_list):
        text = pred_data['text']
        span_list = pred_data['spans']
        index = pred_data['index']
        
        new_span_list = []
        if len(span_list) >= 1:
            for idx, span in enumerate(span_list):
                start = span['start']
                end = span['end']
                label = span['label']
                
                if 'B▲' in label:
                    if idx > 0:
                        # 중간에 새로운 값이 나왔을때
                        new_label = pre_label.split('▲')[-1]
                        new_start, new_end = start_end_adjust(text, new_start, new_end)
                        new_span = {'start':new_start, 'end':new_end, 'label':new_label}
                        new_span_list.append(new_span)
                    
                    new_start = start
                    new_end = end
                    pre_label = label
                    
                if 'I▲' in label:
                    if idx == 0:
                        new_start = start
                        pre_label = label
                    new_end = end
                
            new_label = pre_label.split('▲')[-1]
            new_start, new_end = start_end_adjust(text, new_start, new_end)
            new_span = {'start':new_start, 'end':new_end, 'label':new_label}
            new_span_list.append(new_span)
        
        new_pred_data = {'text':text, 'spans':new_span_list, 'index':index}
        print(new_pred_data)
        new_whole_data_list.append(new_pred_data)
    
    # # 저장            
    # acc_path = f"{root_path}/result_ner/{data_for_analysis.split('.')[0]}" 
    # os.makedirs(acc_path, exist_ok=True)
    # with open(f"{acc_path}/{data_for_analysis.split('.')[0]}_ner_result.jsonl", "w", encoding='utf-8') as f:
    #     for entry in new_whole_data_list:
    #         json_string = json.dumps(entry, ensure_ascii=False)
    #         f.write(json_string+"\n")
    # info = {
    #     'label2id_dict':con_label2id_dict,
    #     'id2label_dict':con_id2label_dict,
    # }
    # with open(f'{acc_path}/args.json', 'w', encoding='utf-8-sig') as f:
    #     json.dump(info, f, indent='\t', ensure_ascii=False)
    
        
if __name__  == '__main__':
    main()