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
    dataset_path = f'{root_path}/datasets/{dataset_name}'
    with open(f'{dataset_path}/data.json', 'r', encoding='utf-8-sig') as f:
        json_dataset = json.load(f)
    label2id_dict = json_dataset['dict']['label2id']
    id2label_dict = json_dataset['dict']['id2label']
    
    if not test_data:
        # 학습시 활용한 데이터셋의 validation 데이터 활용
        print(111111111111111)
        sys.exit()
        test_string_split_list = json_dataset['data']['validation']['string']
    else:
        # 직접 지정한 데이터 활용
        try:
            with open(f'./test_data/{test_data}', 'r', encoding='utf-8-sig') as f:
                test_data_list = json.load(f)
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
        
        # ==
        # test_idx_list = [11, 12]
        # test_data_list = [
        #     {
        #         'text':"현재 의 문제점 은 기술 을 갖추고  있는 국군 이 만들지 못  하는 스텔스 를 드론 으로 만들어 형상 화  하는 기술 습득  하는 방식 의 기술 발전 을 도모 하여야  합니다 . 핵시설 도 핵원료 의 힘 이 아닌 의힘 으로 일어나는 전기 시설 입 니다 . 스텔스 도 풍선 기구 와 같은 배관 공학 과 기계 설비 와 화학 적 가 압 과 팽창 과 공기 유입 과 배기 의 모터 를 중점 으로 증기 기관 의 지구 과학 의 주변 환경 적 과학 입 니다 . 저희 가 증기 다음 에 발전 된 것 이 전기 이며 기계 와 화학 적 연료 가 발전 된 인류 과학 입 니다 . 저희 는 아직도 기초 과학 을 토대 로 살 고  있는 정도 입 니다 . 정팔 위성 GPS 와 레이저 도 레이더 전자 장비 도 을 이용 한 주변 과학 입 니다 . 증기 이후 의 기관 이 발전 되고 화학 적 연료 가 발전 되고 공압 기압 수압 유압 등등 기초 과학 을 토대 로 주변 과학 과 함께 남든 기술 이 많 습니다 . 저 < 장현 철 > 이어뢰 기뢰 미사일 등등 의 제안 사항 도 물속 에서는 공기 와 대기압 과 물 을 이용 한 과학 무기 를 제시  하였지만 그또한 과학 적 기초 를 토대 로 주변 과학 을 토대 로 만든 국방 무기 체계 입 니다 . 공기 중 에서는 공기 와 가 압 과 배기 의 힘 을 이용 한 기초 과학 으로서 주변 과학 을 토대 로 만든 제안 입 니다 . 저 는 아직 인류 는 주면 지구 과학 의 을 많이 이용 한 다고 봅 니다 . 참고 사항 으로 인류 기술 중 에 열의 가열 을 가 하여 가 압  하며 대기압 과 냉각 에 의 해서 식혀서 제품 을 만드는 기술 이 많 습니다 . TV 와 인터넷 으로 비오는 날항공 모함 에서 착륙 당시 에 미국 의 스텔스 기 종 을 볼당시 에 이륙 착륙 시 에 공기 를 빨 아 들이 면서 배기 하고 더워진 공기 를 추진 체가열 로서 유지 시키는 것 을 보았 으며 상부 도 따뜻한 열기로서 가열 증기 를 보았 으며 하부 의 공기 는 따뜻한 공기 와 추진 체 의 연료 소모 의 추진 체연소 가 유지 되는 것 을 보았 으며 추진시에 뒤 로 도 따뜻한 공기 가 배기 되는 증기 상황 을 보았 으며 배관 의 공학 으로서 공기 가 모터 를 통  하여 전달 되어 열선 과 기관 열 로서 유지 되는 사항 이 보입 니다 . 저희 도 국방력 기술 발전 을 도모 하여야  하며 우방 국 의 기술 을 심층 깊게 관찰  하며 대한민국 국방력 을 강화 하여야  하는 사항 이 많 다고 보며 유인기 가 어려우면 무인기 를 만들어야 한 다고 봅 니다 . 따뜻한 공기 가 모터 를 통  하며 기관 과 추진 체 와 함께 강 하게 배관 을 통  하여 증기 배기  하는 것 을 보 았습니다 . 대한민국 도 국방력 기술 발전 을 도모 하여야  하며 유인 비행기 만들기 가 어려우면 무인 비행기 드론 스텔스 를 만들면 된 다고 봅 니다 . 모터 로 내부 배관 구 동시 에 배관 이 점점 좁아 지고 공기 가 강 하게 배기 되면 끝부분 에서 불 이 나 는 현상 이  있는 부하 상태 의 저항 열값 이 생기며 공기 와 배관 은 가열 되는 상황 의 스트레스 를 많이 받는 상황 이 됩 니다 . 소방차 의 강한 물세기 에 소방 호수 도 끝부분 에서 강 하게 분사 되면 들리는 현상 이  있습니다 . 공기 압력 도 같은 원리 로 생각 하여야  하며 공기 가압 과 배관 의 저항 값등등 의 저항 값 에 의 하여 외부 와 공기 가 마찰 이 생겨나서 스트레스 로서 불 이 나 는 상황 을 감안 하여야  합니다 . 기체 를 들어 올릴 정도 의 가열 공기 로서 추진 체 와 함께 뜨거운 바람 을 유지 시키고서 배관 을 통과  하여 배기  하는 바람 의 세기 면 된 다고 봅 니다 . 비행 체 가 항력 을  하고  있는 동안 에 추진 력 을 막아 서면 스텔스 기능 이 되는 사항 이며 진공 부력 이 발생  하여 기체 를 들어 올리는 효과 가  있습니다 .",
        #     },
        #     {
        #         'text':"현재 의 문제점 은 기술 을 갖추고  있는 국군 이 만들지 못  하는 스텔스 를 드론 으로 만들어 형상 화  하는 기술 습득  하는 방식 의 기술 발전 을 도모 하여야  합니다 . 핵시설 도 핵원료 의 힘 이 아닌 의힘 으로 일어나는 전기 시설 입 니다 . 스텔스 도 풍선 기구 와 같은 배관 공학 과 기계 설비 와 화학 적 가 압 과 팽창 과 공기 유입 과 배기 의 모터 를 중점 으로 증기 기관 의 지구 과학 의 주변 환경 적 과학 입 니다 . 저희 가 증기 다음 에 발전 된 것 이 전기 이며 기계 와 화학 적 연료 가 발전 된 인류 과학 입 니다 . 저희 는 아직도 기초 과학 을 토대 로 살 고  있는 정도 입 니다 . 정팔 위성 GPS 와 레이저 도 레이더 전자 장비 도 을 이용 한 주변 과학 입 니다 . 증기 이후 의 기관 이 발전 되고 화학 적 연료 가 발전 되고 공압 기압 수압 유압 등등 기초 과학 을 토대 로 주변 과학 과 함께 남든 기술 이 많 습니다 . 저 < 장현 철 > 이어뢰 기뢰 미사일 등등 의 제안 사항 도 물속 에서는 공기 와 대기압 과 물 을 이용 한 과학 무기 를 제시  하였지만 그또한 과학 적 기초 를 토대 로 주변 과학 을 토대 로 만든 국방 무기 체계 입 니다 . 공기 중 에서는 공기 와 가 압 과 배기 의 힘 을 이용 한 기초 과학 으로서 주변 과학 을 토대 로 만든 제안 입 니다 . 저 는 아직 인류 는 주면 지구 과학 의 을 많이 이용 한 다고 봅 니다 . 참고 사항 으로 인류 기술 중 에 열의 가열 을 가 하여 가 압  하며 대기압 과 냉각 에 의 해서 식혀서 제품 을 만드는 기술 이 많 습니다 . TV 와 인터넷 으로 비오는 날항공 모함 에서 착륙 당시 에 미국 의 스텔스 기 종 을 볼당시 에 이륙 착륙 시 에 공기 를 빨 아 들이 면서 배기 하고 더워진 공기 를 추진 체가열 로서 유지 시키는 것 을 보았 으며 상부 도 따뜻한 열기로서 가열 증기 를 보았 으며 하부 의 공기 는 따뜻한 공기 와 추진 체 의 연료 소모 의 추진 체연소 가 유지 되는 것 을 보았 으며 추진시에 뒤 로 도 따뜻한 공기 가 배기 되는 증기 상황 을 보았 으며 배관 의 공학 으로서 공기 가 모터 를 통  하여 전달 되어 열선 과 기관 열 로서 유지 되는 사항 이 보입 니다 . 저희 도 국방력 기술 발전 을 도모 하여야  하며 우방 국 의 기술 을 심층 깊게 관찰  하며 대한민국 국방력 을 강화 하여야  하는 사항 이 많 다고 보며 유인기 가 어려우면 무인기 를 만들어야 한 다고 봅 니다 . 따뜻한 공기 가 모터 를 통  하며 기관 과 추진 체 와 함께 강 하게 배관 을 통  하여 증기 배기  하는 것 을 보 았습니다 . 대한민국 도 국방력 기술 발전 을 도모 하여야  하며 유인 비행기 만들기 가 어려우면 무인 비행기 드론 스텔스 를 만들면 된 다고 봅 니다 . 모터 로 내부 배관 구 동시 에 배관 이 점점 좁아 지고 공기 가 강 하게 배기 되면 끝부분 에서 불 이 나 는 현상 이  있는 부하 상태 의 저항 열값 이 생기며 공기 와 배관 은 가열 되는 상황 의 스트레스 를 많이 받는 상황 이 됩 니다 . 소방차 의 강한 물세기 에 소방 호수 도 끝부분 에서 강 하게 분사 되면 들리는 현상 이  있습니다 . 공기 압력 도 같은 원리 로 생각 하여야  하며 공기 가압 과 배관 의 저항 값등등 의 저항 값 에 의 하여 외부 와 공기 가 마찰 이 생겨나서 스트레스 로서 불 이 나 는 상황 을 감안 하여야  합니다 . 기체 를 들어 올릴 정도 의 가열 공기 로서 추진 체 와 함께 뜨거운 바람 을 유지 시키고서 배관 을 통과  하여 배기  하는 바람 의 세기 면 된 다고 봅 니다 . 비행 체 가 항력 을  하고  있는 동안 에 추진 력 을 막아 서면 스텔스 기능 이 되는 사항 이며 진공 부력 이 발생  하여 기체 를 들어 올리는 효과 가  있습니다 ."
        #     }
        # ]
        # ==
        
        # 전처리        
        remove_re_df = pd.read_csv('/home/kimyh/python/project/kca/KoBERT-NER/whole_module/supports/remove_re_df.csv', encoding='utf-8-sig')
        remove_string_df = pd.read_csv('/home/kimyh/python/project/kca/KoBERT-NER/whole_module/supports/remove_string_df.csv', encoding='utf-8-sig')
        test_string_list = []
        for test_data in test_data_list:
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
            test_string_split = test_string.split(' ')
            test_string_split_list.append(test_string_split)
        # 너무 긴 문장 제어    
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
    model_path = 'monologg/kobert'
    tokenizer = KoBertTokenizer.from_pretrained(model_path)
    
    # ================================================================================
    num_labels = len(label2id_dict)
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    
    # val dataloader
    # def string_split(string):
    #     result = string.split(' ')
    #     return result 
        
    # whole_string_split_list = list(map(string_split, test_string_list))
    if dummy_label:
        whole_id_split_list = []
        for string_split_list in test_string_split_list:
            id_split_list = []
            for _ in string_split_list:
                id_split_list.append(0)
            whole_id_split_list.append(id_split_list)
    else:
        whole_id_split_list = json_dataset['data']['validation']['id']
    
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
    config_class, model_class = BertConfig, BertForTokenClassification

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
        for ss, tl, pl in zip(ss_list, tl_list, pl_list):
            whole_ss_list.append(ss)
            whole_tl_list.append(tl)
            whole_pl_list.append(pl)
    
    # confusion matrix
    confusion_matrix = clm.make_confusion_matrix(
        mode='label2id', 
        true_list=whole_tl_list, 
        pred_list=whole_pl_list, 
        label2id_dict=label2id_dict
    )
    
    print('데이터 생성중..')
    whole_data_list = []
    for ss_list, pl_list, fi in zip(test_string_split_list, whole_pred_label_list, new_test_idx_list):
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
    for whole_data, b_e in zip(whole_data_list, b_e_list):
        each_data = {"text": None, "spans": [], 'index':None}
        
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
            # print(temp_text)
            # temp_index = whole_data['index']
            if b_e == 'e':
                each_data['text'] = front_text
                each_data['index'] = whole_data['index']
                front_text = ''
                new_whole_data_list.append(each_data)
        else:
            new_whole_data_list.append(whole_data)
    
    for new_whole_data in new_whole_data_list:
        text = new_whole_data['text']
        span_list = new_whole_data['spans']        
        for span in span_list:
            start = span['start']
            end = span['end']
            label = span['label']
        
    print('데이터 저장중..')
    with open(f"./test_result/pred_20231129_{len(new_whole_data_list)}_data_i.jsonl", "w", encoding='utf-8') as jsonl_file:
        for entry in new_whole_data_list:
            json_string = json.dumps(entry, ensure_ascii=False)
            jsonl_file.write(json_string+"\n")
            
    # {"text": "", "spans": [{"start": 102, "end": 106, "label": "상품"}, {"start": 130, "end": 134, "label": "상품"}], "index": 49}
    
        
if __name__  == '__main__':
    main()