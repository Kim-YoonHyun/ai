import sys
import torch.nn as nn
import torch.nn.functional as F
from models import attentionuitls as atm
from models import ffnnutils as ffm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p, activation="relu"):
        super(DecoderLayer, self).__init__()
        
        # self.self_attention = self_attention
        self.multi_head_attention1 = atm.MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads
        )
        self.look_ahead_mask = None
        
        self.multi_head_attention2 = atm.MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads
        )
        self.encoder_decoder_mask = None
        
        self.ffnn = ffm.FeedForwardNeuralNetwork(
            d_model=d_model,
            dropout_p=dropout_p,
            activation=activation
        )
        # self.cross_attention = cross_attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)
        # self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, y, encoder_result, look_ahead_mask, encoder_decoder_mask):
        
        # masking attention
        mha1_y, _ = self.multi_head_attention1(
            y, y, y,
            mask=look_ahead_mask
        )
        mha1_y = self.dropout(mha1_y)
        y = y + mha1_y
        y = self.norm1(y)
        
        # encoder-decoder attention
        mha2_y, _ = self.multi_head_attention2(
            y, encoder_result, encoder_result,
            mask=encoder_decoder_mask
        )
        mha2_y = self.dropout(mha2_y)
        y = y + mha2_y

        # normalize
        y = self.norm2(y)
        
        ffnn_y = self.ffnn(y)
        ffnn_y = self.dropout(ffnn_y)
        
        # add
        y = y + ffnn_y
        
        # normalize
        y = self.norm3(y)
        
        return y


class Decoder(nn.Module):
    def __init__(self, d_layer_num, d_model, n_heads, dropout_p, activation):
        super(Decoder, self).__init__()
        
        decoder_layer_list = []
        for _ in range(d_layer_num):
            Decoder_Layer = DecoderLayer(
                d_model=d_model,
                n_heads=n_heads, 
                dropout_p=dropout_p, 
                activation=activation
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