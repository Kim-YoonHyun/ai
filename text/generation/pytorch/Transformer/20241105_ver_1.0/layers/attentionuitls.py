import math
import numpy as np

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # self.scale = scale
        
    def forward(self, query, key, value, mask=None):
        
        # get attention score
        _, _, _, d_k = query.shape
        key_t = key.transpose(-2, -1)
        # batch_size × n_heads × max_len × d_k
        # --> 
        # batch_size × n_heads × d_k × max_len
        w = torch.matmul(query, key_t)
        attention_score = w / math.sqrt(d_k)
        
        # masking
        if mask is not None:
            attention_score.masked_fill_(mask, -np.inf)
        
        # softmax
        attention_score = torch.softmax(attention_score, dim=-1)

        # value attention
        attention_result = torch.matmul(attention_score, value)
        
        # result
        attention_result = attention_result.contiguous()
        
        return attention_result, attention_score

        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        bs, q_max_len, d_model = query.shape
        _, k_max_len, _ = key.shape
        d_k = d_model // self.n_heads
        
        # query
        query = self.q_linear(query)
        query = query.view(bs, q_max_len, self.n_heads, d_k).transpose(1, 2)
        # view : reshape 와 동일함.
        
        # key
        key = self.k_linear(key)
        key = key.view(bs, k_max_len, self.n_heads, d_k).transpose(1, 2)
        
        # value
        value = self.v_linear(value)
        value = value.view(bs, k_max_len, self.n_heads, d_k).transpose(1, 2)
        
        # attention
        out, attn = self.attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.reshape(bs, q_max_len, d_model)
         
        # linear
        out = self.out_linear(out)
        
        return out, attn