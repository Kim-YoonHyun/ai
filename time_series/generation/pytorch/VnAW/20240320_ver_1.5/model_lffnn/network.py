import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
from layers import embedutils as embm
from layers import encoderutils as encm
from layers import decoderutils as decm
from layers import ffnnutils as ffum


class VnAW(nn.Module):
    def __init__(self, pred_len, embed_type, d_model, d_ff, n_heads, temporal_type,
                 encoder_layer_num, encoder_feature_length, encoder_temporal_length, encoder_activation,
                 decoder_layer_num, decoder_feature_length, decoder_temporal_length, decoder_activation,
                 dropout_p, output_length):
        super(VnAW, self).__init__()
        
        self.pred_len = pred_len
        self.dropout_p = dropout_p
        
        # Embedding
        if embed_type == 0:
            self.enc_embedding = embm.DataEmbedding(
                feature_length=encoder_feature_length, 
                d_model=d_model, 
                max_len=self.pred_len,
                temporal_type=temporal_type, 
                temporal_length=encoder_temporal_length, 
                dropout_p=self.dropout_p
            )
            self.dec_embedding = embm.DataEmbedding(
                feature_length=decoder_feature_length,
                d_model=d_model,
                max_len=self.pred_len,
                temporal_type=temporal_type,
                temporal_length=decoder_temporal_length, 
                dropout_p=dropout_p
            )
        
        # # Encoder
        # self.encoder = encm.Encoder(
        #     layer_num=encoder_layer_num, 
        #     d_model=d_model, 
        #     d_ff=d_ff,
        #     n_heads=n_heads, 
        #     activation=encoder_activation,
        #     dropout_p=dropout_p
        # )
        # self.norm1 = nn.LayerNorm(d_model)
        
        # # Decoder
        # self.decoder = decm.Decoder(
        #     layer_num=decoder_layer_num,
        #     d_model=d_model, 
        #     d_ff=d_ff,
        #     n_heads=n_heads, 
        #     activation=decoder_activation,
        #     dropout_p=dropout_p
            
        # )
        self.lffnn = ffum.LinearFeedForwardNeuralNetwork(d_model, output_length, dropout_p)
    
    
    def forward(self, x, x_mark, y, y_mark,
                enc_self_mask=None, look_ahead_mask=None, enc_dec_mask=None):

        # encoder embedding
        enc_emb = self.enc_embedding(x, x_mark)
        
        # # encoder
        # enc_out, _ = self.encoder(
        #     x=enc_emb, 
        #     enc_self_mask=enc_self_mask
        # )
        
        # # decoder embedding
        # dec_emb = self.dec_embedding(y, y_mark)
        
        # # decoder
        # dec_out = self.decoder(
        #     y=dec_emb, 
        #     encoder_result=enc_out,
        #     look_ahead_mask=look_ahead_mask,
        #     encoder_decoder_mask=enc_dec_mask,
        # )

        # projection
        result = self.lffnn(enc_emb)
        # ==
        # sys.path.append('/home/kimyh/python')
        # from sharemodule import utils
        # utils.save_tensor(result, mode=1)
        # sys.exit()
        # ==
        return result
        
        