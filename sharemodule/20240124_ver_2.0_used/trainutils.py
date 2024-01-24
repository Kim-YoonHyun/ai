import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
# from transformers import AdamW, get_linear_schedule_with_warmup

def see_device():
    '''
    선택 가능한 gpu device 표시
    '''
    return_string = ''
    if torch.cuda.is_available():
        print('\n------------- GPU list -------------')
        n_devices = torch.cuda.device_count()
        for i in range(n_devices):
            print(f'{i}: {torch.cuda.get_device_name(i)}')
        print('------------------------------------')
        
        return_string += f'{i}: {torch.cuda.get_device_name(i)}\n'
    else:
        return_string = 'No GPU available'
    return return_string
        

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
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.is_available():
        device = torch.device(f"cuda")
    else:
        print('No GPU available, using the CPU instead.')
        print('wait 5 second...')
        time.sleep(5)
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
    if base == 'torch':
        if method == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif method == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif method == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:
            raise ValueError('Not a valid optimizer')
    # optimizer = optimizer(params=model.parameters(), lr=learning_rate)
    return optimizer


def get_scheduler(base, method, optimizer, 
                  gamma=0.97, total_iter=None, warmup_iter=None):
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
    if base == 'torch':
        if method == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                1.0,
                gamma=gamma
            )
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
        if method == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                total_iter-warmup_iter
            )
    if base == 'transformers':
        if method == 'linear_schedule_with_warmup':
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_iter, 
                num_training_steps=total_iter
            )
    return scheduler