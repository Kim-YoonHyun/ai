# import torch
# import torch.nn as nn


# class Attention():
#     def __init__(self):
#         super().__init__()
#         self.softmax = nn.Softmax(dim=-1)


#     def forward(self, Q, K, V, mask=None, dk=64):
#         w = torch.bmm(Q, K.transpose(1, 2))
#         if mask:
#             assert w.size() == mask.size()
#             w.masked_fill_(mask, float('-inf'))
#         w = self.softmax(w / (dk**0.5))
#         c = torch.bmm(w, V)

#         return c


# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, n_splits):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.n_splits = n_splits
#         self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.attention = Attention()

    
#     def forward(self, Q, K, V, mask=None):
#         num = self.hidden_size // self.n_splits
#         QWs = self.Q_linear(Q).split(num, dim= -1)
#         KWs = self.Q_linear(K).split(num, dim= -1)
#         VWs = self.Q_linear(V).split(num, dim= -1)

#         QWs = torch.cat(QWs, dim=0)
#         KWs = torch.cat(KWs, dim=0)
#         VWs = torch.cat(VWs, dim=0)

#         if mask:
#             mask_list = []
#             for _ in range(self.n_splits):
#                 mask_list.append(mask)
#             mask = torch.cat(mask_list, dim=0)
        
#         c = self.attention(
#             Q=QWs,
#             K=KWs,
#             V=VWs,
#             mask=mask,
#             dk=num
#         )
#         c = c.split(Q.size(0), dim=0)
#         c = self.linear(torch.cat(c, dim=0-1))
#         return c
    

# class MySequential(nn.Sequential):
#     def forward(self, *layer_list):
#         for module in self._modules.values():
#             layer_list = module(*layer_list)
#         return layer_list


# class EncoderBlock(nn.Module):
#     def __init__(self, hidden_size, n_splits, 
#                  dropout_p=0.1, use_leaky_relu=False):
#         super().__init__()

#         self.Multi_Head_Attention = MultiHeadAttention(hidden_size, n_splits)
#         self.norm_1 = nn.LayerNorm(hidden_size)
#         self.dropout_1 = nn.Dropout(dropout_p)

#         self.fcn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size*4),
#             nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
#             nn.Linear(hidden_size*4, hidden_size)
#         )
#         self.norm_2 = nn.LayerNorm(hidden_size)
#         self.dropout_2 = nn.Dropout(dropout_p)


#     def forawrd(self, x, mask):
#         z = self.norm_1(x)
#         m_h_att = self.Multi_Head_Attention(z, z, z, mask=mask)
#         m_h_att = self.dropout_1(m_h_att)
#         z = x + m_h_att

#         fcn = self.norm_2(z)
#         fcn = self.fcn(fcn)
#         fcn = self.dropout_2(fcn)
#         z = z + fcn

#         return z, mask


# class DecoderBlock(nn.Module):
#     def __init__(self, hidden_size, n_splits, dropout_p=0.1, use_leaky_relu=False):
#         super().__init__()

#         self.Multi_Head_Attention_1 = MultiHeadAttention(hidden_size, n_splits)
#         self.norm_1 = nn.LayerNorm(hidden_size)
#         self.dropout_1 = nn.Dropout(dropout_p)

#         self.Multi_Head_Attention_2 = MultiHeadAttention(hidden_size, n_splits)
#         self.norm_2 = nn.LayerNorm(hidden_size)
#         self.dropout_2 = nn.Dropout(dropout_p)

#         self.fcn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size*4),
#             nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
#             nn.Linear(hidden_size*4, hidden_size)
#         )
#         self.norm_3 = nn.LayerNorm(hidden_size)
#         self.dropout_3 = nn.Dropout(dropout_p)
    
#     def forward(self, x, key_and_value, mask, pre, future_mask):
#         if not pre:
#             z = self.norm_1(x)
#             m_h_att = self.Multi_Head_Attention_1(z, z, z, mask=future_mask)
#             m_h_att = self.dropout_1(m_h_att)
#         else:
#             pre_z = self.norm_1(pre)
#             z = self.norm_1(x)
#             m_h_att = self.Multi_Head_Attention_1(z, pre_z, pre_z, mask=None)
#             m_h_att = self.dropout_1(m_h_att)
#         z = z + m_h_att
        
#         z = self.norm_2(z)
#         norm_key_and_value = self.norm_2(key_and_value)
#         key = norm_key_and_value
#         value = norm_key_and_value
#         m_h_att = self.Multi_Head_Attention_2(z, key, value, mask=mask)
#         m_h_att = self.dropout_2(m_h_att)
#         z = z + m_h_att
        
#         fcn_norm = self.norm_3(z)
#         fcn_norm = self.fcn(fcn_norm)
#         fcn_norm = self.dropout_3(fcn_norm)
#         z = z + fcn_norm
        
#         return z, key_and_value, mask, pre, future_mask
        
        

# class Transformer(nn.Module):
#     def __init__(
#             self, input_size, hidden_size, output_size,
#             n_splits, n_enc_blocks=6, n_dec_blocks=6,
#             dropout_p=0.1, use_leaky_relu=False, max_length=512
#         ):

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_splits = n_splits
#         self.n_enc_blocks = n_enc_blocks
#         self.n_dec_blocks = n_dec_blocks
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         super().__init__()

#         # Embedding
#         self.emb_enc = nn.Embedding(input_size, hidden_size)
#         self.emb_dec = nn.Embedding(output_size, hidden_size)
#         self.emb_dropout = nn.Dropout(dropout_p)

#         # ?
#         self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

#         # encoder
#         encoder_layer_list = []
#         for _ in range(n_enc_blocks):
#             Encoder_Block = EncoderBlock(
#                 hidden_size=hidden_size, 
#                 n_splits=n_splits, 
#                 dropout_p=dropout_p, 
#                 use_leaky_relu=use_leaky_relu
#             )
#             encoder_layer_list.append(Encoder_Block)
#         self.encoder = MySequential(*encoder_layer_list)

#         # decoder
#         decoder_layer_list = []
#         for _ in range(n_dec_blocks):
#             Decoder_Block = DecoderBlock(
#                 hidden_size=hidden_size, 
#                 n_splits=n_splits, 
#                 dropout_p=dropout_p, 
#                 use_leaky_relu=use_leaky_relu
#             )
#             decoder_layer_list.append(Decoder_Block)
#         self.decoder = MySequential(*decoder_layer_list)
        
#         # generator
#         self.generator = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, output_size),
#             nn.LogSoftmax(dim=-1)
#         )
    
#     @torch.no_grad()
#     def _generate_pos_enc(self, hidden_size, max_length):
#         enc = torch.FloatTensor(max_length, hidden_size).zero_()
#         pos = torch.arange(0, max_length).unsqueeze(-1).float()
#         dim = torch.arange(0, hidden_size//2).unsqueeze(0).float()

#         enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
#         enc[:, 1::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
#         return enc
    
#     @torch.no_grad()
#     def _generate_mask(self, x, length):
#         mask_list = []
#         max_length = max(length)
#         for l in length:
#             zero_mat = x.new_ones(1, l).zero_()
#             if l < max_length:
#                 one_mat = x.new_ones(1, (max_length-1))
#                 mask = torch.cat(zero_mat, one_mat, dim=-1)
#             else:
#                 mask = zero_mat
#             mask_list.append(mask)
#         mask = torch.cat(mask_list, dim=0).bool()
#         return mask
        
    
#     def _position_encoding(self, x, init_pos=0):
#         assert x.size(-1) == self.pos_enc.size(-1)
        
#     def forward(self, x, y):
#         with torch.no_grad():
#             mask = self._generate_mask(
#                 x=x[0], 
#                 length=x[1]
#             )
#             encoder_mask = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
#             decoder_mask = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            
#         encoder_embedding = self.emb_enc(x)
#         encoder_embedding = self._position_encoding(encoder_embedding)
#         z = self.emb_dropout(encoder_embedding)