import sys
import torch.nn as nn
from models import embedutils as embm
from models import encoderutils as encm
from models import decoderutils as decm


class VnAW(nn.Module):
    def __init__(self, pred_len, embed_type,
                 enc_in, dec_in, d_model, temporal_type, freq, dropout_p,
                 e_layer_num, factor, n_heads,
                 d_layer_num, activation, c_out):
        super(VnAW, self).__init__()
        
        self.pred_len = pred_len
        self.dropout_p = dropout_p
        # self.get_attention_matrix = get_attention_matrix
        
        # Embedding
        if embed_type == 0:
            self.enc_embedding = embm.DataEmbedding(
                c_in=enc_in, 
                d_model=d_model, 
                max_len=self.pred_len,
                temporal_type=temporal_type, 
                freq=freq, 
                dropout_p=self.dropout_p
            )
            self.dec_embedding = embm.DataEmbedding(
                c_in=dec_in,
                d_model=d_model,
                max_len=self.pred_len,
                temporal_type=temporal_type,
                freq=freq,
                dropout_p=dropout_p
            )
        
        # Encoder
        self.encoder = encm.Encoder(
            e_layer_num=e_layer_num, 
            d_model=d_model, 
            n_heads=n_heads, 
            dropout_p=dropout_p
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # Decoder
        self.decoder = decm.Decoder(
            d_layer_num=d_layer_num,
            d_model=d_model, 
            n_heads=n_heads, 
            dropout_p=dropout_p, 
            activation=activation
        )
        self.norm2 = nn.LayerNorm(d_model)
        # self.linear = nn.LayerNorm(d_model, c_out, bias=True)
        # elementwise_affine = True : 학습가능한 scale 및 shift 파라미터를 가지게되어 
        # 해당 파라미터를 학습하도록 허용하는 것
        # bias = True 에 해당함
        self.linear = nn.LayerNorm(d_model, c_out, elementwise_affine=True)
        # 
    
    
    def forward(self, enc_x, enc_x_mark, dec_x, dec_x_mark,
                enc_self_mask=None, look_ahead_mask=None, enc_dec_mask=None):

        # encoder embedding
        enc_emb = self.enc_embedding(enc_x, enc_x_mark)
        
        # encoder
        enc_out, enc_attention_score_list = self.encoder(
            x=enc_emb, 
            enc_self_mask=enc_self_mask
        )
        
        # normalize
        enc_out_norm = self.norm1(enc_out)
        
        # decoder embedding
        dec_emb = self.dec_embedding(dec_x, dec_x_mark)
        
        # decoder
        dec_out = self.decoder(
            x=dec_emb, 
            cross=enc_out_norm,
            mask=dec_self_mask,
            cross_mask=enc_dec_mask,
        )
        
        # normalize
        dec_out_norm = self.norm2(dec_out)
        
        # projection
        dec_out_lin = self.linear(dec_out_norm)
        
        return dec_out_lin[:, -self.pred_len:, :], enc_attention_score_list
        
        