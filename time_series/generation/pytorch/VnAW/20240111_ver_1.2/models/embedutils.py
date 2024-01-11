import sys
import torch

import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        if torch.__version__ >= '1.5.0':
            padding = 1
        else:
            padding = 2
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular', 
            bias=False
        )
        
        # Conv1d 모듈의 가중치를 kaiming normal 방법으로 초기화
        # leaky relu 에 맞는 초기화를 수행
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.require_grad = False

        # position
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # div term
        base = torch.tensor(10000).float()
        term = torch.arange(0, d_model, 2).float()
        div_term = base.pow((term / d_model))
        
        # positional embedding
        pos_emb[:, 0::2] = torch.sin(position / div_term)
        pos_emb[:, 1::2] = torch.cos(position / div_term)
        pos_emb = pos_emb.unsqueeze(0)
        
        # 학습되지 않는 파라미터 등록
        self.register_buffer('pos_emb', pos_emb)
        
    def forward(self):
        return self.pos_emb
    

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq=6):
        super(TimeFeatureEmbedding, self).__init__()
        self.time_feature_embed = nn.Linear(freq, d_model, bias=False)
    def forward(self, x):
        x = self.time_feature_embed(x)
        return x
    

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len, temporal_type='fixed', freq='h', dropout_p=0.1):
        super(DataEmbedding, self).__init__()

        # token embedding
        self.token_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # positional embeding
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        
        # time feature embedding
        if temporal_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            pass
        
        # dropout            
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, x_mark):
        x = self.token_embedding(x) + self.position_embedding() + self.temporal_embedding(x_mark)
        x = self.dropout(x)
        return x


# ==================================================================







class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x







class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
