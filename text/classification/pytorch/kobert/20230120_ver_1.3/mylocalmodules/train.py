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
from sharemodule import trainutils as tutils


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
        학습시 loss 를 계산할 loss function
    
    optimizer: optimizer
        학습 optimizer. 지정하지 않을 경우 평가모델로 변경

    schduler: scheduler
        학습 learning rate scheduler
    
    amp: int

    max_grad_norm: int
        학습 그래디언트 클리핑 기울기 값

    returns
    -------
    output: float torch.tensor
        예측 결과값이 포함된 tensor

    loss: float torch.tensor
        평가 결과 loss 값

    acc: float numpy array
        평가 결과 정확도 값
    '''
    # loss, 결과 초기화
    loss_sum = 0
    pred_label_list = []
    pred_reliability_list = []
    pred_2nd_label_list = []

    # batch 입력
    for batch_idx, (b_string_ids, b_attention_mask, b_segment_ids, b_label) in enumerate(tqdm(dataloader)):
        # 각 변수를 device 에 올리기
        b_string_ids = b_string_ids.long().to(device)
        b_attention_mask = b_attention_mask.long().to(device)
        b_segment_ids = b_segment_ids.long().to(device)
        b_label = b_label.long().to(device)

        # 모델에 데이터 입력
        pred = model(
            b_string_ids,
            token_type_ids=None, 
            attention_mask=b_attention_mask, 
            labels=b_label
        )
        pred = pred[1]

        # loss 계산
        if loss_function:
            b_label = b_label.to(device, dtype=torch.long)
            loss = loss_function(pred, b_label)
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
          uni_class_list, device, loss_function, optimizer, scheduler, amp, max_grad_norm, reset_class, model_save_path):
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

    batch_size: int
        데이터를 나눌 batch size

    train_dataloader
        학습용 train data 로 이루어진 dataloader
    
    validation_dataloader
        학습시 확인에 활용할 validation data로 이루어진 dataloader

    uni_class_list: str list, shape=(n, )
        데이터의 고유 클래스 list

    device
    - type:
    - description: 학습 진행시 활용할 device (cpu or gpu)

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
    best_val_pred_label_ary: int list, shape=(n, )
        입력된 데이터에 대한 결과(예측) 값 리트스

    model: model
        학습된 model

    history: json dict
        학습 이력을 저장한 json 형식의 dictionary
    '''

   
    # 변수 초기화
    best_acc = 0
    history = {'best':{'epoch':0, 'loss':0, 'acc':0}}

    # 학습 epoch 진행
    start = time.time()
    for epoch in range(epochs):
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

        # validation
        model.eval()
        with torch.no_grad():
            val_pred_label_ary, _, _, val_loss = get_output(
                mode='val',
                model=model, 
                dataloader=validation_dataloader,
                device=device,
                loss_function=loss_function
            )
            val_label_list = validation_dataloader.dataset.label_list
            val_confusion_matrix = tutils.make_confusion_matrix(
                uni_class_list=uni_class_list,
                true=val_label_list,
                pred=val_pred_label_ary,
                reset_class=reset_class
            )
            val_acc = val_confusion_matrix['accuracy'].values[0]
        scheduler.step()  
        
        # 학습 이력 저장
        history[f'epoch {epoch+1}']['train_loss'] = train_loss
        history[f'epoch {epoch+1}']['val_loss'] = val_loss
        history[f'epoch {epoch+1}']['val_acc'] = val_acc

        # 최적의 학습 값 저장 (정확도 기준)
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
            
            # best result save
            best_model_name = f'epoch{str(best_epoch).zfill(4)}'
            os.makedirs(f'{model_save_path}/{best_model_name}', exist_ok=True)
            torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')
            best_val_confusion_matrix.to_csv(f'{model_save_path}/{best_model_name}/confusion_matrix.csv', encoding='utf-8-sig')
           
        print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')
        print(val_confusion_matrix)

        # train history save 
        with open(f'{model_save_path}/train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent='\t', ensure_ascii=False)
        
        h, m, s = utils.time_measure(start)
        print(f'걸린시간: {h}시간 {m}분 {s}초')

    # last result save
    os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')
    val_confusion_matrix.to_csv(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/confusion_matrix.csv', encoding='utf-8-sig')

    # 최적의 학습 모델 불러오기
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