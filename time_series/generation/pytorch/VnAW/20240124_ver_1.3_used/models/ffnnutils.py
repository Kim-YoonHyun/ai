import sys
# import torch
import torch.nn as nn
import torch.nn.functional as F


# import numpy as np

# from math import sqrt
# from models import attentionuitls as atm
# from utils.masking import TriangularCausalMask, ProbMask
# import os


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p, activation):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    
    def forward(self, x):
        # position-wise FFNN
        x = x.transpose(-1, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(-1, 1)
        return x
        