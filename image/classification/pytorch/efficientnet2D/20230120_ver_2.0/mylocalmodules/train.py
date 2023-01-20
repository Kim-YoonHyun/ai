import numpy as np
import sys
import copy
import os
import time
import pandas as pd
import json
from tqdm import tqdm
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
from sharemodule import trainutils as tutils

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
    

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
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
         # ==
        # if batch_idx > 10:
        #     continue
        # ==

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




def train(model, start_epoch, epochs, train_dataloader, validation_dataloader, 
          class_list, device, loss_function, optimizer, scheduler, amp, max_grad_norm, reset_class, model_save_path):

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



    # 최적의 acc 값 초기화
    best_acc = 0
    
    # 학습 이력 초기화
    history = {'best':{'epoch':0, 'loss':0, 'acc':0}}
    start_epoch -= 1

    # 학습 epoch 진행
    start = time.time()
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
            val_confusion_matrix = tutils.make_confusion_matrix(
                uni_class_list=class_list,
                true=val_label_ary,
                pred=val_pred_label_ary,
                reset_class=reset_class
            )
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
            best_val_confusion_matrix.to_csv(f'{model_save_path}/{best_model_name}/confusion_matrix.csv', encoding='utf-8-sig')

        print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')
        print(val_confusion_matrix)

        # train history save 
        with open(f'{model_save_path}/train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent='\t', ensure_ascii=False)

        h, m, s = utils.time_measure(start)
        print(f'걸린시간: {h}시간 {m}분 {s}초')

    # last epoch save
    os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')
    best_val_confusion_matrix.to_csv(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/confusion_matrix.csv', encoding='utf-8-sig')

    # 최적의 학습모델 불러오기
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)

    
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
    # model test
    model.eval()
    with torch.no_grad():
        pred_label_ary, pred_reliability_ary, pred_2nd_label_ary, _ = get_output(
            mode='test',
            model=model, 
            dataloader=test_dataloader,
            device=device,
            loss_function=loss_function
        )
    return pred_label_ary, pred_reliability_ary, pred_2nd_label_ary    
    
    
    
    
    
    
    
