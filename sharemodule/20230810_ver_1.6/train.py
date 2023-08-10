import numpy as np
import shutil
import sys
import copy
import os
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch

# local modules
from mylocalmodules import iterator as im

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils
from sharemodule import classificationutils as cfum


def get_output(mode, purpose, dataloader, model, device, loss_function=None, optimizer=None, scheduler=None, max_grad_norm=None):
    '''
    모델에 데이터를 입력하여 설정에 따른 결과 loss 값을 얻어내는 함수
    optimizer 유무에 따라 학습, 평가 모드로 활용 가능

    parameters
    ----------
    mode: str ('train', 'val', 'test')
        모델에 대해 학습 모드인지 아닌지 설정.

    model: model
        데이터를 입력할 model

    device: gpu or cpu
        학습을 진행할 장치

    loss_function: loss_function
        loss 계산에 적용할 loss function
    
    optimizer: optimizer
        학습 optimizer. 지정하지 않을 경우 평가모델로 변경

    schduler: scheduler
        학습 learning rate scheduler
    
    max_grad_norm: int
        학습 그래디언트 클리핑 기울기 값

    returns
    -------

    '''
    # loss, 결과 초기화
    pred_list = []
    b_label_list = []
    loss_sum = 0
    result_list = []

    # classification 결과 초기화
    if purpose == 'classification':
        pred_label_list = []
        pred_reliability_list = []
        pred_2nd_label_list = []

    # 이터레이터 생성
    # 재사용이 불가능하므로 할때마다 생성 필요
    print('iterator 생성')
    iterator = im.Iterator(
        dataloader=dataloader,
        model=model,
        device=device
    )
    iter_n = 0
    
    # 이터레이터 진행
    for pred, b_label in tqdm(iterator):
        
        # Loss 계산
        if loss_function is not None:
            loss = loss_function(pred, b_label)
            loss_sum += loss.item()
        
        # 모드 선택
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        elif mode in ['val', 'test']:
            pass
        else:
            raise ValueError('Mode should be either train, val, or test')

        # 결과값 저장
        pred = pred.to('cpu').detach().numpy() # 메모리 해제
        pred_list.append(pred)

        # label 값 저장
        b_label_list.append(b_label)
        del b_label

        if purpose == 'classification':
            # 1순위 및 신뢰도
            max_index = np.argmax(pred, axis=1)
            pred_label_list.append(max_index)
            
            # 2순위 및 신뢰도
            max_2nd_index, pred_reliability = cfum.get_max_2nd_n_reliability(pred)
            pred_2nd_label_list.append(max_2nd_index)
            pred_reliability_list.append(pred_reliability)
            
        iter_n += 1

    # 결과 변수
    try:
        pred_ary = np.concatenate(pred_list, axis=0)
        b_label_ary = torch.concat(b_label_list).to('cpu').detach().numpy()
    except KeyError:
        print('에러가 발생하였습니다.')
        sys.exit()
    
    if purpose == 'classification':
        pred_label_ary = np.concatenate(pred_label_list)
        pred_reliability_ary = np.concatenate(pred_reliability_list)
        pred_2nd_label_ary = np.concatenate(pred_2nd_label_list)
        result_list.append(pred_label_ary)
        result_list.append(pred_reliability_ary)
        result_list.append(pred_2nd_label_ary)
    
    running_loss = loss_sum / iter_n
    del iterator
    torch.cuda.empty_cache()
    return pred_ary, b_label_ary, running_loss, result_list


def train(model, purpose, start_epoch, epochs, train_dataloader, validation_dataloader, 
          uni_class_list, device, loss_function, optimizer, scheduler, max_grad_norm, reset_class, model_save_path):
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
    # best_acc = 0
    best_loss = float('inf')
    history = {'best':{'epoch':0, 'loss':0}}

    # 학습 진행
    whole_start = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'val_loss':0}
        print(f'======== {epoch+1:2d}/{epochs} ========')
        print("lr: ", optimizer.param_groups[0]['lr'])

        # train
        model.train()
        _, _, train_loss, _ = get_output(
            mode='train',
            purpose=purpose,
            dataloader=train_dataloader,
            model=model,
            device=device, 
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm
        )
        print(f"epoch {epoch+1} train loss {train_loss}")

        # validation
        model.eval()
        with torch.no_grad():
            val_pred_label_ary, val_true_label_ary, val_loss, _ = get_output(
                mode='val',
                model=model, 
                purpose=purpose,
                dataloader=validation_dataloader,
                device=device,
                loss_function=loss_function,
            )
            if purpose == 'classification':
                # true
                val_true_label_ary = np.reshape(val_true_label_ary, (-1))
                
                # pred
                val_pred_label_ary = np.argmax(val_pred_label_ary, axis=-1)
                val_pred_label_ary = np.reshape(val_pred_label_ary, (-1))
                
                # confusion matrix
                val_confusion_matrix = cfum.make_confusion_matrix(
                    uni_class_list=uni_class_list,
                    true=val_true_label_ary,
                    pred=val_pred_label_ary,
                    reset_class=reset_class
                )
                
                # accuracy
                val_acc = val_confusion_matrix['accuracy'].values[0]
        try:
            scheduler.step() 
        except TypeError:
            scheduler.step(epoch)

        # 학습 이력 저장
        history[f'epoch {epoch+1}']['train_loss'] = train_loss
        history[f'epoch {epoch+1}']['val_loss'] = val_loss

        # 최적의 모델 변수 저장
        if val_loss <= best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            history['best']['epoch'] = best_epoch
            history['best']['loss'] = best_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_dic = {
                "epoch" : epoch,
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "state_dict" : best_model_wts,
                "hyper_parameters" : None,
                "parameter_num" : None
            }

            # 최적 모델 저장
            best_model_name = f'epoch{str(best_epoch).zfill(4)}'
            os.makedirs(f'{model_save_path}/{best_model_name}', exist_ok=True)
            torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')
            
            # 이전의 최적 모델 삭제
            for i in range(epoch, 0, -1):
                pre_best_model_name = f'epoch{str(i).zfill(4)}'
                try:
                    shutil.rmtree(f'{model_save_path}/{pre_best_model_name}')
                    print(f'이전 모델 {pre_best_model_name} 삭제')
                except FileNotFoundError:
                    pass
            
            # 분류 모델인 경우
            if purpose == 'classification':
                best_acc = val_acc
                best_val_confusion_matrix = val_confusion_matrix
                best_val_pred_label_ary = val_pred_label_ary
                history['best']['acc'] = best_acc
                best_val_confusion_matrix.to_csv(f'{model_save_path}/{best_model_name}/confusion_matrix.csv', encoding='utf-8-sig')
                
        # 학습 history 저장
        with open(f'{model_save_path}/train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent='\t', ensure_ascii=False)

        # 검증 결과 출력
        print(f'epoch {epoch+1} validation loss {val_loss}\n')
        if purpose == 'classification':
            print(val_confusion_matrix)

        # 시간 출력(epoch 당)
        h, m, s = utils.time_measure(epoch_start)
        print(f'에포크 시간: {h}시간 {m}분 {s}초\n')
    
    # 시간 출력(전체)
    h, m, s = utils.time_measure(whole_start)
    print(f'전체 시간: {h}시간 {m}분 {s}초\n')

    # 마지막 결과 저장
    os.makedirs(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}', exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/weight.pt')
    
    # 분류 모델인 경우 추가 저장
    if purpose == 'classification':
        val_confusion_matrix.to_csv(f'{model_save_path}/epoch{str(epoch + 1).zfill(4)}/confusion_matrix.csv', encoding='utf-8-sig')

    # 최적의 학습모델 불러오기
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)

    return save_dic


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
        pred_label_ary, pred_reliability_ary, pred_2nd_label_ary, true_label_ary, _ = get_output(
            mode='test',
            model=model, 
            dataloader=test_dataloader,
            device=device,
            loss_function=loss_function,
        )

    return pred_label_ary, pred_reliability_ary, pred_2nd_label_ary, true_label_ary
