import sys
import os
import torch.nn as nn



def get_loss_function(base, method, pad_idx=None):
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
    if base == 'torch':
        if method == 'MSE':
            loss_function = nn.MSELoss()
        if method == 'CrossEntropyLoss':
            if pad_idx is not None:
                loss_function = nn.CrossEntropyLoss(ignore_index=pad_idx)
            else:
                loss_function = nn.CrossEntropyLoss()
    return loss_function


class LossFunction(nn.Module):
    def __init__(self, base, method, pad_idx=None):
        super().__init__()
        self.loss_function = get_loss_function(
            base=base,
            method=method,
            pad_idx=pad_idx
        )
    
    def forward(self, pred_loss, b_label):
        loss = self.loss_function(pred_loss, b_label)
        return loss
    
    

