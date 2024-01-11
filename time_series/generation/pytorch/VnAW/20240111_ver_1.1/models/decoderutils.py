import sys
import torch.nn as nn
import torch.nn.functional as F
from models import attentionuitls as atm
from models import ffnnutils as ffm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = 4 * d_model
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

    def forward(self, x, cross, x_mask, cross_mask):
        
        # 
        new_x, _ = self.multi_head_attention1(
            x, x, x,
            mask=x_mask
        )
        new_x = self.dropout(new_x)
        x = x + new_x
        x = self.norm1(x)
        
        # x = x + self.dropout(self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask
        # )[0])
        # x = self.norm1(x)

        # cross
        x_cross, _ = self.multi_head_attention2(
            x, cross, cross,
            mask=cross_mask
        )
        x_cross = self.dropout(x_cross)
        x = x + x_cross
        # x = x + self.dropout(self.cross_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0])

        y = x = self.norm2(x)
        
        ffnn_x = self.ffnn(y)
        ffnn_x = self.dropout(ffnn_x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # add
        x = x + ffnn_x
        
        # normalize
        x = self.norm3(x)
        # return self.norm3(x + y)
        
        return x


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

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for decoder_layer in self.decoder_layer_list:
            x = decoder_layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        return x