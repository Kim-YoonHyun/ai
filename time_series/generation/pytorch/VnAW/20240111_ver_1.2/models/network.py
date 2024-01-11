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
        self.linear = nn.Linear(d_model, c_out, bias=True)
    
    
    def forward(self, x, x_mark, y, y_mark,
                enc_self_mask=None, look_ahead_mask=None, enc_dec_mask=None):

        # encoder embedding
        enc_emb = self.enc_embedding(x, x_mark)
        
        # encoder
        enc_out, _ = self.encoder(
            x=enc_emb, 
            enc_self_mask=enc_self_mask
        )
        
        # # normalize
        # enc_out = self.norm1(enc_out)
        
        # decoder embedding
        dec_emb = self.dec_embedding(y, y_mark)
        
        # decoder
        dec_out = self.decoder(
            y=dec_emb, 
            encoder_result=enc_out,
            look_ahead_mask=look_ahead_mask,
            encoder_decoder_mask=enc_dec_mask,
        )
        
        # # normalize
        # dec_out = self.norm2(dec_out)
        
        # projection
        result = self.linear(dec_out)
        # ==
        sys.path.append('/home/kimyh/python')
        from sharemodule import utils
        import numpy as np
        import pandas as pd
        x_ary = utils.tensor2array(x_tensor=result)
        b = x_ary[0]
        b = np.round(b, 3)
        df = pd.DataFrame(b)
        df.to_csv(f'./temp.csv', index=False, encoding='utf-8-sig')
        print(df)
        print(x_ary.shape)
        
        # ary = x_ary[0]
        # i, j, k = ary.shape
        # print(i, j, k)
        # for idx in range(k):
        #     a = np.squeeze(ary[:, :, idx:idx+1])
        #     a = np.round(a, 3)
        #     df = pd.DataFrame(a)
        #     df.to_csv(f'./temp{idx}.csv', index=False, encoding='utf-8-sig')
        #     print(df)
        # print(x_ary.shape)
        sys.exit()
        # ==
        
        return result
        
        