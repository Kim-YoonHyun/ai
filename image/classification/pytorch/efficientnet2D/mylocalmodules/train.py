import numpy as np
import sys
import copy
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils

def see_device(logger=None):
    '''
    선택 가능한 gpu device 표시
    '''
    import torch
    if torch.cuda.is_available():
        print('\n------------- GPU list -------------')
        n_devices = torch.cuda.device_count()
        for i in range(n_devices):
            print(f'{i}: {torch.cuda.get_device_name(i)}')
            if logger:
                logger.info(f'{i}: {torch.cuda.get_device_name(i)}')
        print('------------------------------------')
    else:
        print('No GPU available')   
        if logger:
            logger.info('No GPU available')


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
    import os
    import torch
    
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
        기반이 되는 모듈 이름 설정. 'transformers' or 'torch'.

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
            from transformers import AdamW
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    if base == 'torch':
        import torch.optim as optim

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

    #     from torch.nn import functional as F
    
    #     if loss_name == 'crossentropy':
    #         loss_function = F.cross_entropy

    from torch import nn
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
            from transformers.optimization import get_cosine_schedule_with_warmup
            warmup_step = int(t_total * warmup_ratio)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    if base == 'torch':
        import torch.optim as optim

        if method == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        if method == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)
        if method == 'LambdaLR':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                    lr_lambda=lambda epoch:gamma ** epoch,
                                    last_epoch=-1,
                                    verbose=False)
    return scheduler
    

def get_output(mode, model, dataloader, device, loss_function=None, optimizer=None, scheduler=None, amp=False, max_grad_norm=None):
    '''
    dataloader 에서 각 배치의 데이터를 추출하여 모델에 입력, 결과를 얻어내고 모델 파라미터를 학습시키는 함수

    parameters
    ----------
    mode: str
        모델이 학습 되는 상황인지 검증되는 상황인지 정하는 값. train 일경우 학습, val 또는 test 인 경우 검증.

    model: model
        학습 또는 검증할 model
    
    dataloader: dataloader
        모델에 입력할 데이터로 구축되어있는 dataloader
    
    device: device
        학습을 진행할 컴퓨터 장치
    
    loss_function: loss_function
        학습 결과를 활용해서 loss 를 계산할 loss 함수. 설정하지 않을시 계산하지 않음.

    optimizer: optimizer
        학습용 opimizer. 설정하지 않을 시 적용되지 않음.

    scheduler: scheduler
        learning rate 조절용 scheduler. 설정하지 않을 시 적용되지 않음.

    amp: int
        설정하지 않을 시 적용되지 않음

    max_grad_norm: int
        학습시 적용할 크래디언트 클리핑 기울기 값. 설정하지 않을 시 적용되지 않음.
    
    returns
    -------
    pred_label_ary: int list, shape=(n, )
        데이터를 모델에 입력시 계산(예측) 되는 라벨 값 리스트.
    
    running_loss: float
        예측된 라벨 값 리스트와 실제 라벨 값 간의 loss 값.

    '''
    
    # loss, 결과 초기화
    loss_sum = 0
    pred_label_list = []
    pred_reliability_list = []
    pred_2nd_label_list = []
    
    for x, y in tqdm(dataloader):
        # dataloader 내 각 변수값 지정
        x = x.to(device, dtype=torch.float)
        
        # 모델에 데이터 입력
        pred = model(x)

        # Loss 계산
        if loss_function:
            y = y.to(device, dtype=torch.long)
            loss = loss_function(pred, y)
            loss_sum += loss.item()
        
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

        # 결과값 저장
        pred = pred.to('cpu').detach().numpy()
        pred_label_list.append(np.argmax(pred, axis=-1))

        # ==============
        # 신뢰점수 구하기(가칭)
        pred_min = np.expand_dims(np.min(pred, axis=-1), axis=-1)
        pred = pred - pred_min
        pred_max = np.expand_dims(np.max(pred, axis=-1), axis=-1)
        pred = pred/pred_max

        # 1순위 예측값 없애기
        pred = np.where(pred == 1, -100, pred)

        # 2순위 예측값 저장
        pred_2nd_label_list.append(np.argmax(pred, axis=-1))

        # 신뢰도 구하기
        pred_2nd_max = (1 - np.max(pred, axis=-1))*100

        # 신뢰도 저장
        pred_reliability_list.append(pred_2nd_max)
    
    pred_label_ary = np.concatenate(pred_label_list)
    pred_reliability_ary = np.concatenate(pred_reliability_list)
    pred_2nd_label_ary = np.concatenate(pred_2nd_label_list)
    running_loss = loss_sum / len(dataloader)

    return pred_label_ary, pred_reliability_ary, pred_2nd_label_ary, running_loss


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
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    
    try:
        true_label_list = true
        pred_label_list = pred
    except TypeError:
        encoder = LabelEncoder()
        true_label_list = encoder.fit_transform(true) 
        pred_label_list = encoder.fit_transform(pred) 
    
    matrix = np.zeros([len(class_list), len(class_list)])
    for t, o in zip(true_label_list, pred_label_list):
        matrix[t][o] += 1
    
    # 결과 json 생성
    acc_report = classification_report(true_label_list, pred_label_list, output_dict=True, target_names=class_list)
    acc_report['matrix'] = matrix.tolist()

    return acc_report
    

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
    import pandas as pd

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    acc_report = make_acc_report(class_list=class_list, true=true, pred=pred)

    flag = 0
    for e, accs in acc_report.items():
        if flag >= len(class_list):
            accuracy_list[0] = accs
            break
        accuracy_list.append(None)
        precision_list.append(accs['precision'])
        recall_list.append(accs['recall'])
        f1_list.append(accs['f1-score'])
        support_list.append(accs['support'])
        flag += 1

    matrix = acc_report['matrix']
    df1 = pd.DataFrame(matrix, index=class_list, columns=class_list)
    df2 = pd.DataFrame([accuracy_list, precision_list, recall_list, f1_list, support_list], columns=class_list, index=['accuracy', 'precision', 'recall', 'f1', 'support']).T
    confusion_matrix = pd.concat([df1, df2], axis=1)
    
    if save:
        confusion_matrix.to_csv(save, encoding='utf-8-sig')
    
    return confusion_matrix





def train(model, start_epoch, epochs, train_dataloader, validation_dataloader, 
          class_list, device, loss_function, optimizer, scheduler, amp, max_grad_norm, model_save_path):

    '''
    각 학습 파라미터를 기준으로 epoch 를 통해 모델을 학습 및 결과를 얻어내는 함수.

    parameters
    ----------
    model: model
        학습을 할 model

    start_epoch: int
        학습을 시작하는 epoch. re-train시 의미를 가지는 parameter
    
    epochs: int
        학습 에포크
    
    train_dataloader: class
        배치에 따른 데이터셋이 구축되어있는 학습용 dataloader
    
    validation_dataloader: class
        배치에 따른 데이터셋이 구축되어있는 검증용 dataloader

    class_list: list
        데이터의 고유 클래스 리스트

    device: device
        학습을 진행할 컴퓨터의 장치

    loss_function: loss_function
        학습 후 loss 를 계산할 때 적용할 loss function
    
    optimizer: optimizer
        학습 시 적용할 optimizer
    
    scheduler: scheduler
        학습 learning rate 를 조정할 scheduler

    amp: int
        학습 파라미터 중 하나

    max_grad_norm: int
        학습 그래디언트의 기울기

    model_save_path: str
        학습 결과를 저장할 경로.

    returns
    -------
        best_val_pred_label_ary: int list
            각 데이터별 예측된 라벨값 리스트
        
        model: model
            학습된 모델
        
        history: json dict
            학습 이력이 저장된 json 형식의 dictionary
    '''

    import torch
    import copy
    import time
    
    import os

    # 최적의 acc 값 초기화
    best_acc = 0
    
    # 학습 이력 초기화
    history = {'best':{'epoch':0, 'loss':0, 'acc':0}}
    start_epoch -= 1

    # 학습 epoch 진행    
    for epoch in range(start_epoch, epochs):
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'train_acc':0, 'val_loss':0, 'val_acc':0}
        print(f'======== {epoch+1:2d}/{epochs} ========')
        print("lr: ", optimizer.param_groups[0]['lr'])

        # train
        model.train()
        _, _, _, train_loss = get_output(
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

        model.eval()
        with torch.no_grad():
            val_pred_label_ary, _, _, val_loss = get_output(
                mode='val',
                model=model, 
                dataloader=validation_dataloader,
                device=device,
                loss_function=loss_function,
                )
            
            val_label_ary = validation_dataloader.dataset.label_list
            val_confusion_matrix = make_confusion_matrix(
                class_list=class_list,
                true=val_label_ary,
                pred=val_pred_label_ary)
            val_acc = val_confusion_matrix['accuracy'].values[0]

        # Update learning rate schedule
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
            best_val_pred_label_ary = val_pred_label_ary
            history['best']['epoch'] = best_epoch
            history['best']['loss'] = best_loss
            history['best']['acc'] = best_acc

            # make save path
            best_model_name = f'epoch{str(best_epoch).zfill(4)}'
            os.makedirs(f'{model_save_path}/{best_model_name}', exist_ok=True)

            # best model weight save
            torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')

            # val confusion matrix save
            utils.save_csv(save_path=f'{model_save_path}/{best_model_name}/confusion_matrix.csv',
                            data_for_save=best_val_confusion_matrix)

                
        print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')
        print(val_confusion_matrix)

        # train history save 
        utils.save_json(save_path=f'{model_save_path}/train_history.json', data_for_save=history)

    # last epoch save
    os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')
    utils.save_csv(save_path=f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/confusion_matrix.csv',
                    data_for_save=best_val_confusion_matrix)

    # 최적의 학습모델 불러오기
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)

    # return best_val_pred_label_ary, model, history
    
    
def model_test(model, test_dataloader, device, loss_function):
    '''
    학습을 진행하여 최적의 학습된 모델 및 학습 이력을 얻어내는 함수

    parameters
    -------------------------------------------------------------
    model
        학습을 진행할 model
    
    test_dataloader
        학습시 확인에 활용할 test data로 이루어진 dataloader

    device
    - type:
    - description: 학습 진행시 활용할 device (cpu or gpu)

    returns
    -------
    pred_label_ary: int list, shape=(n, )
        입력된 데이터에 대한 결과(예측) 값 리tmxm
    '''
    import torch
    import copy
   
    # model test
    model.eval()
    with torch.no_grad():
        pred_label_ary, pred_reliability_ary, pred_2nd_label_ary, _ = get_output(
            mode='test',
            model=model, 
            dataloader=test_dataloader,
            device=device,
            loss_function=loss_function)
    return pred_label_ary, pred_reliability_ary, pred_2nd_label_ary    
    
    
    
    
    
    
    
