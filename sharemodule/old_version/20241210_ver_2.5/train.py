import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import shutil
import copy
import time
import json
import warnings
warnings.filterwarnings('ignore')

import torch

# share modules
sys.path.append('/home/kimyh/python/ai')
# from sharemodule import utils as um
from sharemodule import timeutils as tum
from sharemodule import classificationutils as cfum

# local modules
from mylocalmodules import iterator as im


def train(model, purpose, start_epoch, epochs, train_dataloader, validation_dataloader, 
          device, criterion, optimizer, scheduler, max_grad_norm, model_save_path,
          argmax_axis=1, label2id_dict=None, id2label_dict=None):

    # 변수 초기화
    best_loss = float('inf')
    history = {'best':{'epoch':0, 'loss':0}}

    # 학습 진행
    whole_start = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'val_loss':0}
        print(f'======== {epoch+1:2d}/{epochs} ========')
        print(f'model path : {model_save_path}')
        print("lr: ", optimizer.param_groups[0]['lr'])

        # train -------------------
        model.train()
        
        # get output        
        _, _, train_loss, _ = im.iterator(
            mode='train',
            purpose=purpose,
            dataloader=train_dataloader,
            model=model,
            device=device, 
            criterion=criterion,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm
        )
        print(f"epoch {epoch+1} train loss {train_loss}")

        # validation -------------
        model.eval()
        with torch.no_grad():
            val_pred_label_ary, val_true_label_ary, val_loss, _ = im.iterator(
                mode='val',
                model=model, 
                purpose=purpose,
                dataloader=validation_dataloader,
                device=device,
                criterion=criterion,
            )
            
            
            if purpose == 'classification':
                # true
                val_true_label_ary = np.reshape(val_true_label_ary, (-1))
                
                # pred
                val_pred_label_ary = np.argmax(val_pred_label_ary, axis=argmax_axis)
                val_pred_label_ary = np.reshape(val_pred_label_ary, (-1))
                
                # confusion matrix
                if id2label_dict is not None:
                    val_confusion_matrix = cfum.make_confusion_matrix(
                        mode='id2label',
                        true_list=val_true_label_ary,
                        pred_list=val_pred_label_ary,
                        round_num=4,
                        id2label_dict=id2label_dict
                    )
                if label2id_dict is not None:
                    val_confusion_matrix = cfum.make_confusion_matrix(
                        mode='label2id',
                        true_list=val_true_label_ary,
                        pred_list=val_pred_label_ary,
                        round_num=4,
                        label2id_dict=label2id_dict
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
        h, m, s = tum.time_measure(epoch_start)
        print(f'에포크 시간: {h}시간 {m}분 {s}초\n')
    
    # 시간 출력(전체)
    h, m, s = tum.time_measure(whole_start)
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


def model_test(model, test_dataloader, device):
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
        logit_ary, b_label_ary, running_loss, result_list = im.iterator(
            mode='test',
            purpose='',
            dataloader=test_dataloader,
            model=model, 
            device=device
        )

    return logit_ary, b_label_ary, running_loss, result_list
