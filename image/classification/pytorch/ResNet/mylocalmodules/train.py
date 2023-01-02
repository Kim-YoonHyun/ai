import numpy as np
import sys
import copy
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

from mylocalmodules import utils


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


'''text'''
def get_bert_tokenizer(vocab):
    tokenizer = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    return tokenizer



def get_optimizer(optimizer_name, model, learning_rate):
    
    '''
    학습용 optimizer 를 생성하는 함수.

    parameters
    ----------
    optimizer_name: str
        생성할 optimizer 이름
    
    model: model
        optimizer에 적용할 파라미터값을 불러올 model

    learning_rate: float
        optimizer 에 적용할 learning rate

    returns
    -------
    optimizer: optimizer
        학습에 활용할 optimizer
    '''

    import torch.optim as optim
    if optimizer_name == 'sgd':
        optimizer = optim.SGD
    elif optimizer_name == 'adam':
        optimizer = optim.Adam
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW
    else:
        raise ValueError('Not a valid optimizer')
    
    optimizer = optimizer(params=model.parameters(), lr=learning_rate)

    return optimizer


def get_loss_function(loss_name):
    '''
    학습시 loss 를 계산할 loss function 을 생성하는 함수

    parameters
    ----------
    loss_name: str
        생성할 loss function 의 이름

    returns
    -------
    loss_function: loss function
        학습 loss 를 계산하는 loss function

    '''
    from torch.nn import functional as F
    
    if loss_name == 'crossentropy':
        loss_function = F.cross_entropy
    
    # from torch import nn
    # if function == 'crossentropy':
    #     loss_function = nn.CrossEntropyLoss()

    return loss_function


def get_scheduler(method, optimizer, gamma):
    '''
    학습시 learning rate 를 조정할 scheduler 를 생성하는 함수

    parameters
    ----------
    method: str
        scheduler 이름
    
    optimizer: optimizer
        scheduler 를 적용할 optimizer
    
    gamma: float
        learning rate 를 조정할 비율

    returns
    -------
    scheduler: scheduler
        선택한 이름으로 생성된 scheduler
    '''
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
    
    
    '''image'''
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        # dataloader 내 각 변수값 지정
        x = x.to(device, dtype=torch.float)
        
        # 모델에 데이터 입력
        pred = model(x)

    '''text'''
    # batch 입력
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(dataloader)):
        # dataloader 내 각 변수값 지정
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)

        # 모델에 데이터 입력
        pred = model(token_ids, valid_length, segment_ids)

    '''common'''    
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

        pred = pred.to('cpu').detach().numpy()
        pred_label_list.append(np.argmax(pred, axis=-1))

    pred_label_ary = np.concatenate(pred_label_list)
    running_loss = loss_sum / len(dataloader)

    return pred_label_ary, running_loss



def train(model, start_epoch, epochs, batch_size, train_dataloader, validation_dataloader, class_list, device, loss_function, optimizer, scheduler, amp, max_grad_norm, model_save_path):
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
    
    for epoch in range(start_epoch, epochs):
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'train_acc':0, 'val_loss':0, 'val_acc':0}
        print(f'======== {epoch+1:2d}/{epochs} ========')
        print("lr: ", optimizer.param_groups[0]['lr'])

        # train
        model.train()
        _, train_loss = get_output(
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
            val_pred_label_ary, val_loss = get_output(
                mode='val',
                model=model, 
                dataloader=validation_dataloader,
                device=device,
                loss_function=loss_function,
                )
            
            val_label_ary = validation_dataloader.dataset.label_list
            val_confusion_matrix = utils.make_confusion_matrix(
                class_list=class_list,
                true=val_label_ary,
                pred=val_pred_label_ary)
            val_acc = val_confusion_matrix['accuracy'].values[0]

        scheduler.step() # Update learning rate schedule

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

            # best model weight save
            best_model_name = f'batch{batch_size}_epoch{str(best_epoch).zfill(4)}'
            utils.createfolder(f'{model_save_path}/{best_model_name}')
            torch.save(model.state_dict(), f'{model_save_path}/{best_model_name}/weight.pt')

            # val confusion matrix save
            utils.save_csv(save_path=f'{model_save_path}/{best_model_name}/confusion_matrix.csv',
                            data_for_save=best_val_confusion_matrix)

            # train history save ------------------------------------------------------
            utils.save_json(save_path=f'{model_save_path}/{best_model_name}/train_history.json', data_for_save=history)
                
        print(f'epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.2f}\n')
        print(val_confusion_matrix)
        
    # 최적의 학습모델 불러오기
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)

    return best_val_pred_label_ary, model, history
    
    
    
    
    
    
    
    
    
    
