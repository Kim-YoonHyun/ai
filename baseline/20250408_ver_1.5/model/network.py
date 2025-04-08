import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
from layers import embedutils as embm
from layers import encoderutils as encm
from layers import decoderutils as decm


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # self.conv = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
        self.linear1 = nn.Linear(2, 2, bias=True)
        self.linear2 = nn.Linear(2, 2, bias=True)
        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout_p)
            
    def forward(self, x):
        result = self.linear1(x)
        result = self.linear2(result)
        result = self.activation(result)
            
        return result
    
    
