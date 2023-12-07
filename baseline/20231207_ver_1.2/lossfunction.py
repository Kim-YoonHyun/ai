import sys
import os
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self, input1):
        super().__init__()
        self.input1 = input1
        pass
    
    def forward(self, pred, b_label):
        loss = pred[0]
        return loss
    