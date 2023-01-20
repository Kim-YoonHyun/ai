import numpy as np
import sys
import copy
import os
import time
import pandas as pd
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch import nn

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils


def see_device():
    '''
    선택 가능한 gpu device 표시
    '''
    if torch.cuda.is_available():
        print('\n------------- GPU list -------------')
        n_devices = torch.cuda.device_count()
        for i in range(n_devices):
            print(f'{i}: {torch.cuda.get_device_name(i)}')
        print('------------------------------------')
    else:
        print('No GPU available')   


# gpu or cpu 선택
def get_device(gpu_idx):
    '''
    학습에 활용할 gpu 선택 (없을 시 cpu)

    parameters
    ----------
    gpu_idx: int
        학습에 활용할 gpu 번호(순서)

    returns
    -------
    device: gpu or cpu
        학습에 활용할 gpu or cpu
    '''
    # Arrange GPU devices starting from 0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.is_available():
        device = torch.device(f"cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_optimizer(base, method, model, learning_rate):
    '''
    학습용 optimizer 를 얻어내는 코드
    
    paramaters
    ----------
    base: str
        기반이 되는 모듈 이름 설정. transformers or torch.

    method: str
        optimizer 의 종류 설정.

    model: torch.model
        optimizer 를 적용할 model
    
    learning_rate: float
        learning rate

    returns
    -------
    optimizer: optimizer
        학습용 optimizer
    '''
    if base == 'transformers':
        if method == 'AdamW':
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    if base == 'torch':
        if method == 'sgd':
            optimizer = optim.SGD
        elif method == 'adam':
            optimizer = optim.Adam
        elif method == 'AdamW':
            optimizer = optim.AdamW
        else:
            raise ValueError('Not a valid optimizer')
        optimizer = optimizer(params=model.parameters(), lr=learning_rate)
    return optimizer


def get_loss_function(method):
    '''
    학습시 loss 를 계산할 loss function 을 생성하는 함수

    paramerters
    -----------
    method: str
        생성할 loss function 의 이름

    returns
    -------
    loss_function
    '''
    if method == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()
    return loss_function


def get_scheduler(base, method, optimizer, t_total=0, warmup_ratio=1.0, gamma=0.97):
    '''
    학습용 scheduler 를 얻어내는 함수

    parameters
    ----------
    base: str
        기반이 되는 모듈 이름 설정. transformers or torch.

    method: str
        scheduler 의 종류 설정.

    optimizer: optimizer
        학습용 optimizer

    t_total: float

    warmup_ratio: float

    gamma:float
        learning rate 를 줄이는 비율
    '''
    if base == 'transformers':
        if method == 'cosine_warmup':
            warmup_step = int(t_total * warmup_ratio)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=t_total
            )
        if method == 'linear_warmup':
            warmup_step = int(t_total * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_step,
                num_training_steps=t_total
            )
    if base == 'torch':
        if method == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=gamma
            )
        if method == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=20, 
                verbose=1
            )
        if method == 'LambdaLR':
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda epoch:gamma ** epoch,
                last_epoch=-1,
                verbose=False
            )
    return scheduler


def make_acc_report(uni_class_list, true, pred, reset_class=False):
    '''
    sklearn 의 classification_report 의 결과에 confusion matrix 를 더한 
    json 형태의 결과 데이터를 얻어내는 함수.

    parameters
    ----------
    uni_class_list: str list
        결과를 계산할 class list

    true: int list
        예측 결과 계산에 활용할 true 라벨 데이터

    pred: int list
        예측 결과 계산에 활용할 predict 라벨 데이터

    returns
    -------
    acc_report: json
        각 class 별 정확도 및 정확도 matrix 가 포함된 json 형태의 결과값
    '''
    try:
        true_label_list = true
        pred_label_list = pred
    except TypeError:
        encoder = LabelEncoder()
        true_label_list = encoder.fit_transform(true) 
        pred_label_list = encoder.fit_transform(pred) 

    # matrix
    matrix = np.zeros([len(uni_class_list), len(uni_class_list)])
    for t, o in zip(true_label_list, pred_label_list):
        matrix[t][o] += 1
    
    if reset_class:
        # 각 결과의 유니크 값만 정리
        unique_true_label_list = np.unique(true_label_list)
        unique_pred_label_list = np.unique(pred_label_list)

        # 합집합
        unique_label_list = list(set(unique_true_label_list).union(set(unique_pred_label_list)))

        # 존재하는 라벨값으로만 재구성
        uni_class_ary = np.array(uni_class_list)
        uni_class_list = uni_class_ary[unique_label_list].tolist()

        # 존재하는 라벨값으로만 matrix 재구성    
        matrix = np.array(matrix)
        matrix = matrix[unique_label_list].T
        matrix = matrix[unique_label_list].T

    # 결과 json 생성
    acc_report = classification_report(
        true_label_list, 
        pred_label_list, 
        output_dict=True, 
        target_names=uni_class_list
    )
    acc_report['matrix'] = matrix.tolist()

    return acc_report, uni_class_list


def make_confusion_matrix(uni_class_list, true, pred, reset_class):
    '''
    make_acc_json 함수의 결과 데이터로 pandas DataFrame 기반의 result table 을 만드는 함수.
    경로 설정시 .csv 형태로 저장

    parameters
    ----------
    uni_class_list: str list
        결과를 계산할 class list

    true: int list
        예측 결과 계산에 활용할 true 라벨 데이터

    pred: int list
        예측 결과 계산에 활용할 predict 라벨 데이터

    save: str
        result table 을 저장할 경로 및 이름. default=None (결과저장 X)

    returns
    -------
    confusion_matrix: pandas dataframe, csv
        학습 결과를 가독성이 좋은 형태로 변경한 dataframe. 결과 저장시 csv 로 저장됨.
    '''
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    acc_report, uni_class_list = make_acc_report(
        uni_class_list=uni_class_list, 
        true=true,
        pred=pred,
        reset_class=reset_class
    )
    for e, accs in acc_report.items():
        if e == 'accuracy':
            accuracy_list[0] = accs
            break
        accuracy_list.append(None)
        precision_list.append(accs['precision'])
        recall_list.append(accs['recall'])
        f1_list.append(accs['f1-score'])
        support_list.append(accs['support'])

    matrix = acc_report['matrix']
    df1 = pd.DataFrame(matrix, index=uni_class_list, columns=uni_class_list)
    df2 = pd.DataFrame([accuracy_list, precision_list, recall_list, f1_list, support_list], columns=uni_class_list, index=['accuracy', 'precision', 'recall', 'f1', 'support']).T
    confusion_matrix = pd.concat([df1, df2], axis=1)

    return confusion_matrix









