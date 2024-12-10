import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn

from layers import attentionuitls as atm
from layers import ffnnutils as ffm
from layers import embedutils as ebum


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout_p):
        super(DecoderLayer, self).__init__()
        
        self.masking_attention = atm.MultiHeadAttention(d_model, n_heads)
        self.mha_norm = nn.LayerNorm(d_model)
        
        self.enc_dec_attention = atm.MultiHeadAttention(d_model, n_heads)
        self.ed_norm = nn.LayerNorm(d_model)
        
        self.ffnn = ffm.FFNN(d_model, d_ff, dropout_p)
        self.ff_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, y, enc_out, look_ahead_mask, enc_dec_mask):
        
        # masking attention
        mha_y, _ = self.masking_attention(y, y, y, look_ahead_mask)
        mha_y = self.dropout(mha_y)
        
        # residual
        r1_y = y + mha_y
        
        # normalize
        n1_y = self.mha_norm(r1_y)
        
        # encoder-decoder attention
        ed_y, _ = self.enc_dec_attention(n1_y, enc_out, enc_out, enc_dec_mask)
        ed_y = self.dropout(ed_y)
        
        # residual
        r2_y = n1_y + ed_y
        
        # normalize
        n2_y = self.ed_norm(r2_y)
        
        # Feed-Forward NN
        ffnn_y = self.ffnn(n2_y)
        ffnn_y = self.dropout(ffnn_y)
        
        # residual
        r3_y = n2_y + ffnn_y
        
        # normalize
        n3_y = self.ff_norm(r3_y)
        
        return n3_y


class Decoder(nn.Module):
    def __init__(self, layer_num, max_len, vocab_size, d_model, d_ff, n_heads, dropout_p):
        super(Decoder, self).__init__()
        
        self.embedding = ebum.DataEmbedding(
            max_len=max_len, 
            d_model=d_model,
            vocab_size=vocab_size,
            dropout_p=dropout_p
        )
        
        decoder_layer_list = []
        for _ in range(layer_num):
            Decoder_Layer = DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads, 
                dropout_p=dropout_p
            )
            decoder_layer_list.append(Decoder_Layer)
        self.decoder_layer_list = nn.ModuleList(decoder_layer_list)
        
        self.out_linear = nn.Linear(d_model, vocab_size)

    def forward(self, y, enc_out, look_ahead_mask, enc_dec_mask):
        y = self.embedding(y)
        for decoder_layer in self.decoder_layer_list:
            y = decoder_layer(
                y=y, 
                enc_out=enc_out, 
                look_ahead_mask=look_ahead_mask,
                enc_dec_mask=enc_dec_mask
            )
        y = self.out_linear(y)
        return y