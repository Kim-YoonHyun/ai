'''
https://hi-lu.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EA%B8%B0%EC%B4%88-RNN-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
'''
import numpy as np
import torch.nn as nn


class Tanh(nn.Module):
    def __init__(self, x):
        super(Tanh, self).__init__()
        self.x = x
        
    def forward(self, x):
        exp_m = np.exp(-x)
        exp_p = np.exp(x)
        result = (exp_p - exp_m) / (exp_p + exp_m)
        return result
        
    



class RNN(nn.Module):
    def __init__(self, batch_size, length, embedding_size, hidden_dim, h=None, activation='tanh'):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        
        self.weight_list = []
        self.bias_list = []
        
        if activation == 'tanh':
            self.activation = Tanh()
        # elif activation == 'sigmoid':
        #     self.activation = Sigmoid()
            
        if h is None:
            self.h = np.zeros((self.batch_size, self.length+1, self.hidden_dim))
        else:
            self.h = h

        for _ in range(self.length):
            w_h = np.random.rand(hidden_dim, hidden_dim)
            w_x = np.random.rand(embedding_size, hidden_dim)
            weight = np.concatenate((w_h, w_x), axis=0)
            self.weight_list.append(weight)
            self.bias_list.append(np.random.rand(hidden_dim))


    def forward(self, x):
        x = x[0]
        output_list = []
        for i in range(self.length):
            concat = np.concatenate((self.h[:, i, :], x[:, i, :]), axis=1)
            output = np.matmul(concat, self.weight[i])
            output = output + self.bias[i]
            
            self.h[:, i+1, :] = output
            output_ = self.activation(output)
            output_list.append(output_)
        
        output_ary = np.array(output_list)
        output_ary = np.transpose(output_ary, (1, 0 ,2))
        result = self.h[:, 1:], output_ary
        
