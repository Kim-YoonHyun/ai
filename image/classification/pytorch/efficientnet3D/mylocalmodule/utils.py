import numpy as np
import copy
import time
import sys

import torch

sys.path.append('/home/kimyh/ai')
from myglobalmodule.utils import createfolder

def normalize_3D(data):
    '''
    입력한 3D data를 image 한 장 단위로 normalize 하는 함수

    parameters
    ----------
    data: numpy array
        3D image 데이터 집합체. (data_n, image_n, image_size, image_size)
    
    returns
    -------
    norm_data: numpy array
        same as input data. normalize 된 image 데이터 집합체
    
    '''
    from tqdm import tqdm
    norm_data = []
    for idx, img_3d in enumerate(tqdm(data)):
        norm_data.append([])
        for img in img_3d:
            if np.max(img) != 0.0:
                aver = np.average(img)
                std = np.std(img)
                img = np.divide(np.subtract(img, aver), std)
            norm_data[idx].append(img)
    norm_data = np.array(norm_data)
    return norm_data



# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def get_batch(batch_size, data):
    '''
    학습데이터를 batch 화 시키는 함수

    parameters
    ----------
    batch_size: int
        데이터에 적용할 batch size

    data: numpy array
        batch 를 적용할 데이터. (data_n, image_n, image_size, image_size)

    returns
    -------
    data_b: list
        batch 가 적용된 데이터.(batch_lenght, batch_size, image_n, image_size, image_size)
        batch_lenght X batch_size = data_n.
    '''

    batch_len = len(data) // batch_size

    data_b = []
    for idx in range(batch_len + 1):
        batched = data[batch_size*idx:batch_size*(idx+1)]
        if len(batched) != 0:
            data_b.append(batched)

    return data_b


# calculate the loss per epochs
def get_output(model, loss_function, data, label, device, sanity_check=False, optimizer=None):
    """
    모델에 데이터를 입력하여 얻어낸 결과를 통해 loss 를 구하고
    model weight 를 학습시키는 코드.
    optimizer 유무를 통해 validation 에 활용 가능

    parameters
    ----------
    model: torch model
        학습에 활용할 네트워크 모델

    loss_function: torch.nn.modules.loss.<function>
        학습에 활용할 loss function

    data: numpy array
        batch 화 된 학습 데이터. (batch_lenght, batch_size, ct_image_page_number, img_size, img_size)

    label: numpy array
        batch 화 된 학습 라벨. (batch_length, batch_size)

    device: cuda or cpu
        학습을 진행할 장치

    optimizer: torch.optim.<optimizer>
        학습에 활용할 optimizer

    returns
    -------
    loss: float
        데이터 입력후 계산되어지는 loss 값    
    """
    # loss 변수 초기화
    running_loss = 0.0
    len_data = 0
    
    # 배치별로 결과 계산
    for xb, yb in zip(data, label):
        xb = torch.Tensor(xb)
        yb = torch.Tensor(yb)
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb.type(torch.int64)
        
        output = model(xb)
        loss_b = loss_function(output, yb)
        
        if optimizer:
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
        
        loss_b = loss_b.item()
        running_loss += loss_b
        len_data += len(xb)

        if sanity_check is True:
            break

    # loss 계산
    loss = running_loss/len_data
    return loss


def train(model, params, device):
    '''
    efficient net 을 통해 3d image 를 학습시키는 코드

    parameters
    ----------
    model: torch model
        학습을 진행할 네트워크 모델

    params: dictionary
        학습에 활용할 변수를 모은 dictionary
        ----------
        epochs: int
            학습 에포크 값

        loss_function: torch.nn.modules.loss.<function>
            학습에 활용할 loss function

        optimizer: torch.optim.<optimizer>
            학습에 활용할 optimizer

        train_data: numpy array
            학습용 데이터. (data_n, ct_image_page_number, img_size, img_size)

        train_label: numpy array
            학습 데이터 라벨. (data_n, )

        val_data: numpy array
            학습시 validation 용 데이터. (data_n, ct_image_page_number, img_size, img_size)

        val_label: numpy array
            학습 validation 라벨. (data_n, )

        batch_size: int
            학습 batch

        lr_scheduler: torch.optim.lr_scheduler.<scheduler>
            learning rate 를 조정할 scheduler
        
        model_save_path: str
            모델 및 결과를 저장할 폴더 경로(자동생성)
    
    device: cuda or cpu
        학습을 진행할 장치

    returns
    -------
    model: pytorch model
        최적의 학습 weigth가 적용된 학습 모델
    
    history: json
        각 epoch 별 학습 결과 및 최적의 결과값
    '''

    # 변수 지정
    epochs = params['epochs']
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    train_data = params['train_data']
    train_label = params['train_label']
    val_data = params['val_data']
    val_label = params['val_label']
    batch_size = params['batch_size']
    lr_scheduler = params['lr_scheduler']
    model_save_path = params['model_save_path']

    # 초기 bset loss 값 설정(무한대 값)
    best_loss = float('inf')

    # history dict 초기화
    history = {'best':{'epoch':0, 'loss':0}}
    start_time = time.time()
    
    # batch 분배
    train_data_b = get_batch(batch_size, train_data)
    train_label_b = get_batch(batch_size, train_label)
    val_data_b = get_batch(batch_size, val_data)
    val_label_b = get_batch(batch_size, val_label)

    # epoch 진행
    for epoch in range(epochs):
        current_lr = get_lr(optimizer)
        print(f'Epoch {epoch + 1}/{epochs}, current lr={current_lr:.4f}')
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'val_loss':0}

        # 학습
        model.train()
        train_loss = get_output(model, loss_function, train_data_b, train_label_b, device, optimizer)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = get_output(model, loss_function, val_data_b, val_label_b, device)

        # 최적의 epoch, loss, weight 저장
        if val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            history['best']['epoch'] = best_epoch
            history['best']['loss'] = best_loss
        
        # learning rate 감소(조건부)
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        # 결과 표시 및 이력 저장
        print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, time: {(time.time()-start_time)/60:.4f} min')
        print('-'*10)
        history[f'epoch {epoch+1}']['train_loss'] = train_loss
        history[f'epoch {epoch+1}']['val_loss'] = train_loss

    # 최적의 모델 저장
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)
    createfolder(model_save_path)
    torch.save(model.state_dict(), f'{model_save_path}/batch{batch_size}_epoch{str(best_epoch).zfill(4)}.pt')

    return model, history