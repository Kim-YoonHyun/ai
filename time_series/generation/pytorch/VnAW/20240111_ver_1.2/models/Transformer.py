import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
    
        # Encoder
        Encoder_layer_list = []
        for _ in range(configs.e_layers):
            Full_Attention = FullAttention(
                mask_flag=False, 
                factor=configs.factor, 
                attention_dropout=configs.dropout, 
                output_attention=configs.output_attention
            )
            Attention_Layer = AttentionLayer(
                attention=Full_Attention,
                d_model=configs.d_model, 
                n_heads=configs.n_heads
            )
            Encoder_Layer = EncoderLayer(
                    attention=Attention_Layer,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            Encoder_layer_list.append(Encoder_Layer)
        self.encoder = Encoder(
            attn_layers=Encoder_layer_list,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        # Decoder

        decoder_layer_list = []
        for _ in range(configs.d_layers):
            Full_Attention_self = FullAttention(
                mask_flag=True, 
                factor=configs.factor, 
                attention_dropout=configs.dropout, 
                output_attention=False
            )
            Attention_Layer_self = AttentionLayer(
                attention=Full_Attention_self,
                d_model=configs.d_model, 
                n_heads=configs.n_heads
            )
            Full_Attention_cross = FullAttention(
                mask_flag=True, 
                factor=configs.factor, 
                attention_dropout=configs.dropout, 
                output_attention=False
            )
            Attention_Layer_cross = AttentionLayer(
                attention=Full_Attention_cross,
                d_model=configs.d_model, 
                n_heads=configs.n_heads
            )
            Decoder_Layer = DecoderLayer(
                self_attention=Attention_Layer_self,
                cross_attention=Attention_Layer_cross,
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            )
            decoder_layer_list.append(Decoder_Layer)
        self.decoder = Decoder(
            layers=decoder_layer_list,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
