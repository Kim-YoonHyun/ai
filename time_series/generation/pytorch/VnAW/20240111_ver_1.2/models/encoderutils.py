import sys
# import torch
import torch.nn as nn


# import numpy as np

# from math import sqrt
from models import attentionuitls as atm
from models import ffnnutils as ffm
# from utils.masking import TriangularCausalMask, ProbMask
import os


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, activation, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = atm.MultiHeadAttention(
            d_model=d_model, 
            n_heads=n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffnn = ffm.FeedForwardNeuralNetwork(
            d_model=d_model, 
            d_ff=d_ff, 
            dropout_p=dropout_p, 
            activation=activation
        )
        self.dropout = nn.Dropout(dropout_p)
        
        
    def forward(self, x, enc_self_mask):#, mask_type='self'):
        # multi-head attention
        mha_x, attention_score = self.multi_head_attention(
            x, x, x,
            mask=enc_self_mask
        )
        mha_x = self.dropout(mha_x)
        
        # add
        x = x + mha_x
        
        # normalize
        norm_x = self.norm1(x)
        
        # FFNN        
        ffnn_x = self.ffnn(norm_x)
        ffnn_x = self.dropout(ffnn_x)
        
        # add
        x = norm_x + ffnn_x
        
        # normalize
        x = self.norm2(x)

        return x, attention_score


class Encoder(nn.Module):
    def __init__(self, layer_num, d_model, d_ff, n_heads, activation, dropout_p=0.1):
        super(Encoder, self).__init__()
        encoder_layer_list = []
        for _ in range(layer_num):
            encoder_layer = EncoderLayer(
                d_model=d_model, 
                d_ff=d_ff,
                n_heads=n_heads, 
                activation=activation,
                dropout_p=dropout_p
            )
            encoder_layer_list.append(encoder_layer)
        self.encoder_layer_list = nn.ModuleList(encoder_layer_list)
        
        
    def forward(self, x, enc_self_mask):
        attention_score_list = []
        for encoder_layer in self.encoder_layer_list:
            x, attention_score = encoder_layer(x, enc_self_mask)
            
            attention_score_list.append(attention_score)
        
        return x, attention_score_list

        
# =====================================================================


'''
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

'''