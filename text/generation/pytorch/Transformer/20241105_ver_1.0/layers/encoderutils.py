import sys
import os
sys.path.append(os.getcwd())

import torch.nn as nn

from layers import attentionuitls as atm
from layers import ffnnutils as ffm
from layers import embedutils as ebum


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = atm.MultiHeadAttention(
            d_model=d_model, 
            n_heads=n_heads
        )
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffnn = ffm.FFNN(d_model, d_ff, dropout_p)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)
        
        
    def forward(self, x, enc_self_mask):
        # multi-head attention
        mha_x, _ = self.multi_head_attention(x, x, x, enc_self_mask)
        mha_x = self.dropout(mha_x)
    
        # residual
        x = x + mha_x
        
        # normalize
        norm_x = self.mha_norm(x)
        
        # FFNN        
        ffnn_x = self.ffnn(norm_x)
        ffnn_x = self.dropout(ffnn_x)
        
        # residual
        x = norm_x + ffnn_x
        
        # normalize
        x = self.ff_norm(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, layer_num, max_len, vocab_size, d_model, d_ff, n_heads, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.embedding = ebum.DataEmbedding(max_len, d_model, vocab_size, dropout_p)
        
        encoder_layer_list = []
        for _ in range(layer_num):
            encoder_layer = EncoderLayer(
                d_model=d_model, 
                d_ff=d_ff,
                n_heads=n_heads, 
                dropout_p=dropout_p
            )
            encoder_layer_list.append(encoder_layer)
        self.encoder_layer_list = nn.ModuleList(encoder_layer_list)
        
        
    def forward(self, x, enc_self_mask):
        # embedding
        x = self.embedding(x)
        
        # encoder
        for encoder_layer in self.encoder_layer_list:
            x = encoder_layer(x, enc_self_mask)            
        
        return x

