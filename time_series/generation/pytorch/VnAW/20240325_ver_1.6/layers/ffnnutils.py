import sys
# import torch
import torch.nn as nn
import torch.nn.functional as F


# import numpy as np

# from math import sqrt
# from models import attentionuitls as atm
# from utils.masking import TriangularCausalMask, ProbMask
# import os


class ConvFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p, activation):
        super(ConvFeedForwardNeuralNetwork, self).__init__()
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
        

class LinearFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, d_model, output_length, dropout_p):
        super(LinearFeedForwardNeuralNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, 256, bias=True)
        self.linear2 = nn.Linear(256, 128, bias=True)
        self.linear3 = nn.Linear(128, output_length, bias=True)
        
        # self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        # self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # self.linear3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    
    def forward(self, x):
        # position-wise FFNN
        x = self.linear1(x)
        x = self.activation1(x)
        # x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.activation2(x)
        # x = self.dropout2(x)
        
        x = self.linear3(x)
        return x
    
    
class ResidualLinearFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, d_model, output_length, dropout_p):
        super(ResidualLinearFeedForwardNeuralNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_model, bias=True)
        self.linear2 = nn.Linear(d_model, 256, bias=True)
        self.linear3 = nn.Linear(256, 128, bias=True)
        self.linear4 = nn.Linear(128, 64, bias=True)
        self.linear5 = nn.Linear(64, 32, bias=True)
        self.linear6 = nn.Linear(32, output_length, bias=True)
        
        # self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation4 = nn.ReLU()
        self.activation5 = nn.ReLU()
        # self.dropout1 = nn.Dropout(dropout_p)
        # self.dropout2 = nn.Dropout(dropout_p)
        # self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # self.linear3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
    
    def forward(self, x):
        # position-wise FFNN
        l1 = self.linear1(x)
        l1 = self.activation1(l1)
        x1 = x + l1
        
        x2 = self.linear2(x1)
        x2 = self.activation2(x2)
        
        x3 = self.linear3(x2)
        x3 = self.activation3(x3)
        
        x4 = self.linear4(x3)
        x4 = self.activation4(x4)
        
        x5 = self.linear5(x4)
        x5 = self.activation5(x5)
        
        x6 = self.linear6(x5)
        return x6