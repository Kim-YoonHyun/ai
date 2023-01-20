import numpy as np
import sys
import copy
import os
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch import nn


sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def see_device():
    '''
    선택 가능한 gpu device 표시
    '''
    import torch
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
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Arrange GPU devices starting from 0
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



def get_output(mode, model, dataloader, device, loss_function=None, optimizer=None, scheduler=None, amp=None, max_grad_norm=None):
    '''
    모델에 데이터를 입력하여 설정에 따른 결과 loss 값을 얻어내는 함수
    optimizer 유무에 따라 학습, 평가 모드로 활용 가능

    parameters
    ----------
    mode: str ('train', 'val', 'test')
        모델에 대해 학습 모드인지 아닌지 설정.

    model: model
        데이터를 입력할 model

    dataloader: dataloader
        모델에 입력할 data로 구성된 dataloader

    device: gpu or cpu
        학습을 진행할 장치

    loss_function: loss_function
        loss 계산에 적용할 loss function
    
    optimizer: optimizer
        학습 optimizer. 지정하지 않을 경우 평가모델로 변경

    schduler: scheduler
        학습 learning rate scheduler
    
    amp: int

    max_grad_norm: int
        학습 그래디언트 클리핑 기울기 값

    returns
    -------
    pred_label_ary: numpy array
        예측 라벨값 array

    true_label_ary: numpy array
        실제 라벨값 array
    
    running_loss: float
        학습 loss 값
    '''
    # loss, 결과 초기화
    loss_sum = 0
    pred_label_list = []
    true_label_list = []
    
    # for x, y in enumerate(tqdm(dataloader)):
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        # ==
        # if batch_idx > 1:
        #     continue
        # ==

        # dataloader 내 각 변수값 지정
        x = x.to(device, dtype=torch.float)

        # 모델에 데이터 입력
        pred = model(x)
        pred = pred['out']

        # Loss 계산
        if loss_function:
            y = y.to(device, dtype=torch.long)
            loss = loss_function(pred, y)
            loss_sum += loss.item()
            y = y.to('cpu').detach().numpy()
            true_label_list.append(y)
        
        # 모드 선택
        if mode == 'train':
            optimizer.zero_grad()
            
            # amp 유무
            if amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        elif mode in ['val', 'test']:
            pass
        else:
            raise ValueError('Mode should be either train, val, or test')

        pred = pred.to('cpu').detach().numpy()
        max_index = np.argmax(pred, axis=-3)
        pred_label_list.append(max_index)

    pred_label_ary = np.concatenate(pred_label_list)
    if loss_function:
        true_label_ary = np.concatenate(true_label_list)
    else:
        true_label_ary = true_label_list
    running_loss = loss_sum / len(dataloader)

    return pred_label_ary, true_label_ary, running_loss


def make_acc_report(class_list, true, pred):
    '''
    sklearn 의 classification_report 의 결과에 confusion matrix 를 더한 
    json 형태의 결과 데이터를 얻어내는 함수.

    parameters
    ----------
    class_list: str list
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
    matrix = np.zeros([len(class_list), len(class_list)])
    for t, o in zip(true_label_list, pred_label_list):
        matrix[t][o] += 1
    
    # 각 결과의 유니크 값만 정리
    unique_true_label_list = np.unique(true_label_list)
    unique_pred_label_list = np.unique(pred_label_list)

    # 합집합
    unique_label_list = list(set(unique_true_label_list).union(set(unique_pred_label_list)))

    # 존재하는 라벨값으로만 재구성
    class_list = np.array(class_list)
    class_list = class_list[unique_label_list]

    # 존재하는 라벨값으로만 matrix 재구성    
    matrix = np.array(matrix)
    matrix = matrix[unique_label_list].T
    matrix = matrix[unique_label_list].T

    # 결과 json 생성
    acc_report = classification_report(
        true_label_list, 
        pred_label_list, 
        output_dict=True, 
        target_names=class_list
    )
    acc_report['matrix'] = matrix.tolist()

    return acc_report, class_list
    

def make_confusion_matrix(class_list, true, pred, save=None):
    '''
    make_acc_json 함수의 결과 데이터로 pandas DataFrame 기반의 result table 을 만드는 함수.
    경로 설정시 .csv 형태로 저장

    parameters
    ----------
    class_list: str list
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
    
    acc_report, new_class_list = make_acc_report(
        class_list=class_list, 
        true=true,
        pred=pred
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
    df1 = pd.DataFrame(matrix, index=new_class_list, columns=new_class_list)
    df2 = pd.DataFrame([accuracy_list, precision_list, recall_list, f1_list, support_list], columns=new_class_list, index=['accuracy', 'precision', 'recall', 'f1', 'support']).T
    confusion_matrix = pd.concat([df1, df2], axis=1)

    if save:
        confusion_matrix.to_csv(save, encoding='utf-8-sig')
    
    return confusion_matrix


def train(model, start_epoch, epochs, train_dataloader, validation_dataloader, 
          class_list, device, loss_function, optimizer, scheduler, amp, max_grad_norm, model_save_path):
    '''
    학습을 진행하여 최적의 학습된 모델 및 학습 이력을 얻어내는 함수

    parameters
    -------------------------------------------------------------
    model
        학습을 진행할 model
    
    start_epoch: int
        학습 시작 epoch. re-traine 시 의미를 가짐.

    epochs: int
        학습 epochs 수

    train_dataloader
        학습용 train data 로 이루어진 dataloader
    
    validation_dataloader
        학습시 확인에 활용할 validation data로 이루어진 dataloader

    class_list: str list, shape=(n, )
        데이터의 고유 클래스 list

    device
        학습 진행시 활용할 device (cpu or gpu)

    loss_function: nn.criterion
        학습용 loss_function 

    optimizer 
        학습용 optimizer

    scheduler
        learning rate scheduler

    amp

    max_grad_norm: float        
        그래디언트 클리핑 기울기값

    model_save_path: str
        최종 결과를 저장할 폴더 경로

    returns
    -------
    best_val_pred_label_ary
        입력된 데이터에 대한 결과(예측) 값 array
    true_label_ary: np.array
        실제 라벨값 array
    
    model: model
        학습된 model

    history: json dict
        학습 이력을 저장한 json 형식의 dictionary
    '''

    # 변수 초기화
    best_acc = 0
    history = {'best':{'epoch':0, 'loss':0, 'acc':0}}

    for epoch in range(start_epoch, epochs):
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'train_acc':0, 'val_loss':0, 'val_acc':0}
        print(f'======== {epoch+1:2d}/{epochs} ========')
        print("lr: ", optimizer.param_groups[0]['lr'])

        # train
        model.train()
        _, _, train_loss = get_output(
            mode='train',
            model=model,
            dataloader=train_dataloader,
            device=device, 
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            amp=amp,
            max_grad_norm=max_grad_norm
        )
        print(f"epoch {epoch+1} loss {train_loss:.6f}")

        # validation
        model.eval()
        with torch.no_grad():
            pred_label_ary, true_label_ary, val_loss = get_output(
                mode='val',
                model=model, 
                dataloader=validation_dataloader,
                device=device,
                loss_function=loss_function,
            )
            # 모든 이미지의 pixel confusion matrix
            true_label_ary = np.reshape(true_label_ary, (-1))
            pred_label_ary = np.reshape(pred_label_ary, (-1))
            val_confusion_matrix = make_confusion_matrix(
                class_list=class_list,
                true=true_label_ary,
                pred=pred_label_ary)
            val_acc = val_confusion_matrix['accuracy'].values[0]
        scheduler.step() 

        # 학습 이력 저장
        history[f'epoch {epoch+1}']['train_loss'] = train_loss
        history[f'epoch {epoch+1}']['val_loss'] = val_loss
        history[f'epoch {epoch+1}']['val_acc'] = val_acc

        # 최적의 모델 변수 저장        
        if val_acc > best_acc:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_confusion_matrix = val_confusion_matrix
            best_val_pred_label_ary = pred_label_ary
            history['best']['epoch'] = best_epoch
            history['best']['loss'] = best_loss
            history['best']['acc'] = best_acc
            
            # make save path
            best_model_name = f'epoch{str(best_epoch).zfill(4)}'
            os.makedirs(f'{model_save_path}/{best_model_name}', exist_ok=True)

            # best model weight save
            torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')

            # val confusion matrix save
            best_val_confusion_matrix.to_csv(f'{model_save_path}/{best_model_name}/confusion_matrix.csv', encoding='utf-8-sig')

        # train history save
        with open(f'{model_save_path}/train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent='\t', ensure_ascii=False)

        print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')
        print(val_confusion_matrix)

    # last model weight save
    os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')

    # last confusion matrix save
    val_confusion_matrix.to_csv(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/confusion_matrix.csv', encoding='utf-8-sig')

    # 최적의 학습모델 불러오기
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)

    return best_val_pred_label_ary, true_label_ary, model, history
    

def model_test(model, test_dataloader, device, loss_function):
    '''
    학습을 진행하여 최적의 학습된 모델 및 학습 이력을 얻어내는 함수

    parameters
    -------------------------------------------------------------
    model
        학습을 진행할 model
    
    test_dataloader
        학습용 test data 로 이루어진 dataloader
    
    device
        학습 진행시 활용할 device (cpu or gpu)

    loss_function: nn.criterion
        학습용 loss_function 

    -------
    pred_label_ary: int list, shape=(n, )
        입력된 데이터에 대한 결과(예측) 값 array

    true_label_ary
        실제 라벨 array
    '''
    # validation
    model.eval()
    with torch.no_grad():
        pred_label_ary, true_label_ary, _ = get_output(
            mode='test',
            model=model, 
            dataloader=test_dataloader,
            device=device,
            loss_function=loss_function,
        )

    return pred_label_ary, true_label_ary

    
    
    
    
    
    
    

    
    
    
    
    
