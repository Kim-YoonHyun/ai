import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
from layers import embedutils as embm
from layers import encoderutils as encm
from layers import decoderutils as decm


class VnAW(nn.Module):
    def __init__(self, x_len, y_len, d_model, d_ff, n_heads, 
                 projection_type, embed_type, temporal_type,
                 enc_layer_num, 
                 dec_layer_num,  
                 enc_feature_len, 
                 dec_feature_len, 
                 enc_temporal_len,
                 dec_temporal_len, 
                 enc_act,
                 dec_act,
                 dropout_p, output_length):
        super(VnAW, self).__init__()
        
        # self.pred_len = pred_len
        self.embed_type = embed_type
        self.enc_layer_num = enc_layer_num
        self.dec_layer_num = dec_layer_num
        self.dropout_p = dropout_p
        self.projection_type = projection_type
        
        # Embedding
        if embed_type == 'origin':
            self.enc_embedding = embm.OriginDataEmbedding(
                temporal_type=temporal_type,  max_len=x_len, embed_size=d_model,
                feature_length=enc_feature_len, temporal_length=enc_temporal_len, 
                dropout_p=self.dropout_p
            )
            self.dec_embedding = embm.OriginDataEmbedding(
                temporal_type=temporal_type, max_len=y_len, embed_size=d_model,
                feature_length=dec_feature_len, temporal_length=dec_temporal_len, 
                dropout_p=dropout_p
            )
            
        if embed_type == 'pure':
            self.enc_embedding = embm.PureDataEmbedding(
                max_len=x_len, embed_size=d_model, 
                feature_length=enc_feature_len, 
                dropout_p=self.dropout_p
            )
            self.dec_embedding = embm.PureDataEmbedding(
                embed_size=d_model, max_len=y_len,
                feature_length=dec_feature_len,
                dropout_p=dropout_p
            )
            
        # Encoder
        if enc_layer_num > 0:
            self.encoder = encm.Encoder(
                layer_num=enc_layer_num, 
                d_model=d_model, 
                d_ff=d_ff,
                n_heads=n_heads, 
                activation=enc_act,
                dropout_p=dropout_p
            )
            
        # Decoder
        if dec_layer_num > 0:
            self.decoder = decm.Decoder(
                layer_num=dec_layer_num,
                d_model=d_model, 
                d_ff=d_ff,
                n_heads=n_heads, 
                activation=dec_act,
                dropout_p=dropout_p
            )
            
        if self.projection_type == 'normal':
            self.conv = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout_p)
            self.linear = nn.Linear(d_model, output_length, bias=True)
            
        if self.projection_type == 'res_conv':
            self.conv1 = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
            self.activation2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_p)
            self.linear = nn.Linear(d_model, output_length, bias=True)
            
        if self.projection_type == 'linear':
            self.linear = nn.Linear(d_model, output_length, bias=True)
            
        
        # # self.conv = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
        # # self.activation = nn.ReLU()
        # # self.dropout = nn.Dropout(dropout_p)
        
        # self.conv1 = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
        # self.activation1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=x_len, out_channels=y_len, kernel_size=1)
        # self.activation2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout_p)
        
        # # self.linear = nn.Linear(d_model, output_length, bias=True)
        
        # self.linear0 = nn.Linear(d_model, 256, bias=True)
        # self.linear1 = nn.Linear(256, 128, bias=True)
        # self.linear2 = nn.Linear(128, 64, bias=True)
        # self.linear3 = nn.Linear(64, output_length, bias=True)
    
    
    def forward(self, x, x_mark, y, y_mark,
                enc_self_mask=None, look_ahead_mask=None, enc_dec_mask=None):

        # encoder embedding
        if self.embed_type == 'origin':
            enc_embed = self.enc_embedding(x, x_mark)
        if self.embed_type == 'pure':
            enc_embed = self.enc_embedding(x)
        
        # encoder
        if self.enc_layer_num > 0:
            enc_result, _ = self.encoder(
                x=enc_embed, 
                enc_self_mask=enc_self_mask
            )
        else:
            enc_result = enc_embed
        
        # decoder
        if self.dec_layer_num > 0:
            
            # decoder embedding
            if self.embed_type == 'origin':
                dec_emb = self.dec_embedding(y, y_mark)
            if self.embed_type == 'pure':
                dec_emb = self.dec_embedding(y)

            dec_result = self.decoder(
                y=dec_emb, 
                encoder_result=enc_result,
                look_ahead_mask=look_ahead_mask,
                encoder_decoder_mask=enc_dec_mask,
            )
        else:
            dec_result = enc_result
        
        # projection
        if self.projection_type == 'normal':
            result = self.conv(dec_result)
            result = self.activation(result)
            result = self.dropout(result)
            result = self.linear(result)
            
        if self.projection_type == 'res_conv':
            # residual
            result = self.conv1(dec_result)
            result = self.activation1(result)
            result = self.conv2(result)
            result = result + dec_result
            # projection            
            result = self.activation2(result)
            result = self.dropout2(result)
            result = self.linear(result)
            
            
        if self.projection_type == 'linear':
            result = self.linear(dec_result)
        #==
        # sys.path.append('/home/kimyh/python')
        # from sharemodule import utils
        # utils.save_tensor(result, mode=1)
        # sys.exit()
        #==
        return result
    
    
