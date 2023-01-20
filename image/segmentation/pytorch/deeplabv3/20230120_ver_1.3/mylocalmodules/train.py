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
        # 각 변수를 device 에 올리기
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

    train_dataloader
        학습용 train data 로 이루어진 dataloader
    
    validation_dataloader
        학습시 확인에 활용할 validation data로 이루어진 dataloader

    uni_class_list: str list, shape=(n, )
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

    # 학습 진행
    start = time.time()
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
            val_confusion_matrix = tutils.make_confusion_matrix(
                uni_class_list=uni_class_list,
                true=true_label_ary,
                pred=pred_label_ary,
                reset_class=reset_class
            )
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

    
    
    
    
    
    
    

    
    
    
    
    
