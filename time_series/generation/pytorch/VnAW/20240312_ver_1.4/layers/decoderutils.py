import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn

from layers import attentionuitls as atm
from layers import ffnnutils as ffm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, activation, dropout_p):
        super(DecoderLayer, self).__init__()
        
        self.multi_head_attention1 = atm.MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads
        )
        
        self.multi_head_attention2 = atm.MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads
        )
        
        self.ffnn = ffm.FeedForwardNeuralNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout_p=dropout_p
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, y, encoder_result, look_ahead_mask, encoder_decoder_mask):
        
        # masking attention
        mha1_y, _ = self.multi_head_attention1(
            y, y, y,
            mask=look_ahead_mask
        )
        mha1_y = self.dropout(mha1_y)
        
        # residual
        y = y + mha1_y
        
        # normalize
        y = self.norm1(y)

        # encoder-decoder attention
        mha2_y, _ = self.multi_head_attention2(
            y, encoder_result, encoder_result,
            mask=encoder_decoder_mask
        )
        mha2_y = self.dropout(mha2_y)
        
        # residual
        y = y + mha2_y
        
        # normalize
        y = self.norm2(y)
        
        # Feed-Forward NN
        ffnn_y = self.ffnn(y)
        ffnn_y = self.dropout(ffnn_y)
        y = y + ffnn_y
        
        # normalize
        y = self.norm3(y)
        
        return y


class Decoder(nn.Module):
    def __init__(self, layer_num, d_model, d_ff, n_heads, activation, dropout_p):
        super(Decoder, self).__init__()
        
        decoder_layer_list = []
        for _ in range(layer_num):
            Decoder_Layer = DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads, 
                activation=activation,
                dropout_p=dropout_p
            )
            decoder_layer_list.append(Decoder_Layer)
        self.decoder_layer_list = nn.ModuleList(decoder_layer_list)

    def forward(self, y, encoder_result, look_ahead_mask=None, encoder_decoder_mask=None):
        for decoder_layer in self.decoder_layer_list:
            y = decoder_layer(
                y=y, 
                encoder_result=encoder_result, 
                look_ahead_mask=look_ahead_mask, 
                encoder_decoder_mask=encoder_decoder_mask
            )    
        return y