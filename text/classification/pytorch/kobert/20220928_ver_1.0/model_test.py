'''
install requirement
!pip install numpy
!pip install pandas
!pip install tqdm
!torch install --> https://pytorch.org/get-started/locally/
!pip install mxnet
!pip install gluonnlp
!pip install sentencepiece
!pip install transformers==3.0.2
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
'''

# 일반 module
import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# custom module
sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils
from mylocalmodules import data_loader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils



def main():
    # -------------------------------------------------------------------------
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('phase', help='프로젝트의 진행 단계')
    parser.add_argument('model_name', help='학습된 모델 이름')
    parser.add_argument('condition_order', help='학습 조건 번호')
    parser.add_argument('model_epoch', help='학습된 모델 중 선택할 에포크 진행 정도')
    parser.add_argument('--label_answer', type=bool, default=True, help='정답 클래스가 있는지 여부')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # trained args setting
    with open(f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/args_setting_{args.condition_order}.json', 'r', encoding='utf-8-sig') as f:
        trained_args_setting = json.load(f)
    
    # trained variable
    class_list = trained_args_setting['class_list'].split(',')
    batch_size = trained_args_setting['batch_size']
    max_len = trained_args_setting['max_len']
    num_workers = trained_args_setting['num_workers']
    pin_memory = trained_args_setting['pin_memory']
    drop_last = trained_args_setting['drop_last']
    device_num = trained_args_setting['device_num']
    loss_name = trained_args_setting['loss_name']
    random_seed = trained_args_setting['random_seed']
    
    # -------------------------------------------------------------------------
    # envs setting 
    utils.envs_setting(random_seed)

    # logger
    logger = utils.get_logger('predict', logging_level='info')

    # -------------------------------------------------------------------------
    # get test data
    print('\n>>> get dataset...')
    dataset_name = args.model_name[:-6]
    print(dataset_name)
    with open(f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/{dataset_name}.json', 'r', encoding='utf-8-sig') as f:
        dataset = json.load(f)

    string_list = dataset['validation_data']['string']
    print('데이터 예시:')
    print(string_list[0])
    

    if args.label_answer:
        label_list = dataset['validation_data']['label']
        print(f'라벨: {label_list[0]}')
    else:
        label_list = np.full(len(string_list), 0)

    # -------------------------------------------------------------------------
    # get tokenizer
    print('\n>>> get kobert tokenizer...')
    tokenizer = dutils.get_kobert_tokenizer(pre_trained='skt/kobert-base-v1')
    
    # get vacab
    vocab = dutils.get_vocab(tokenizer)

    # get dataloader
    print('\n>>> make dataloader')
    Test_Dataloader = dutils.get_dataloader(string_list=string_list,
                                           label_list=label_list,
                                           batch_size=batch_size,
                                           tokenizer=tokenizer.tokenize,
                                           vocab=vocab, 
                                           max_len=max_len,
                                           pad=True,
                                           pair=False,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)
    
    # -------------------------------------------------------------------------
    # device selection
    device = tutils.get_device(device_num)

    # model
    result_save_path = f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/{args.model_epoch}'
    model = mutils.get_bert_model(method='BertForSequenceClassification', pre_trained="skt/kobert-base-v1", num_labels=len(class_list))
    model.load_state_dict(torch.load(f'{result_save_path}/weight.pt'))
    model = model.to(device)

    # loss function
    if args.label_answer:
        loss_function = tutils.get_loss_function(method=loss_name)
    else:
        loss_function = None

    # -------------------------------------------------------------------------
    # model test
    pred_label_ary, pred_reliability_ary, pred_2nd_label_ary= tutils.model_test(
        model=model,
        test_dataloader=Test_Dataloader,
        device=device,
        loss_function=loss_function)

    # -------------------------------------------------------------------------
    true_label_list = Test_Dataloader.dataset.label_list
    string_list = Test_Dataloader.dataset.string_list

    if args.label_answer:
        test_confusion_matrix = tutils.make_confusion_matrix(
            class_list=class_list,
            true=true_label_list,
            pred=pred_label_ary)
        utils.save_csv(save_path=f'{result_save_path}/test_confusion_matrix.csv',
                       data_for_save=test_confusion_matrix)

    # result df save
    true_class_list = []
    pred_class_list = []
    pred_2nd_class_list = []
    for true_label, pred_label, pred_2nd_label in zip(true_label_list, pred_label_ary, pred_2nd_label_ary):
        true_class = class_list[true_label]
        pred_class = class_list[pred_label]
        pred_2nd_class = class_list[pred_2nd_label]
        true_class_list.append(true_class)
        pred_class_list.append(pred_class)
        pred_2nd_class_list.append(pred_2nd_class)
    
    result_df = pd.DataFrame([true_class_list, pred_class_list, pred_reliability_ary, pred_2nd_class_list, string_list], index=['true', 'pred', 'reliability', 'pred 2nd', 'string']).T
    print(f"민원: {result_df['string'][0]}")
    print(f"분류: {result_df['true'][0]}")
    print(f"예측: {result_df['pred'][0]}")
    print(f"신뢰도: {result_df['reliability'][0]}")
    print(f"2순위 예측: {result_df['pred 2nd'][0]}")
    



    utils.save_csv(save_path=f'{result_save_path}/test_result_df.csv', 
                   data_for_save=result_df, 
                   index=False)


if __name__ == '__main__':
    main()