import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#==
def temp_df(tensor, name):
    import pandas as pd
    temp_list = tensor.tolist()
    ary = np.round(np.array(temp_list), 2)
    df = pd.DataFrame(ary)
    df.to_csv(f'./{name}.csv', index=False, encoding='utf-8-sig')
#==

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.matmul
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        key_transpose = torch.transpose(key, -2, -1)
        w = torch.matmul(query, key_transpose)
        d_k = query.size()[-1]
        attention_score = w/math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==False, float('-inf'))
        softmax_attention_score = self.softmax(attention_score)
        result = self.matmul(softmax_attention_score, value)
        return result, softmax_attention_score
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8, d_model=512):
        super().__init__()
        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num
        self.attention = Attention()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.linear_3 = nn.Linear(d_model, d_model)
        self.linear_4 = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.linear_1(query)
        query = query.view(batch_size, -1, self.head_num, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.linear_2(key)
        key = key.view(batch_size, -1, self.head_num, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.linear_3(value)
        value = value.view(batch_size, -1, self.head_num, self.d_k)
        value = value.transpose(1, 2)
        
        attention_result, attention_score = self.attention(query, key, value, mask)
        attention_result = attention_result.transpose(1, 2).contiguous()
        attention_result = attention_result.view(batch_size, -1, self.head_num * self.d_k)
        #==
        print(attention_result[0])
        print(attention_result.size())
        temp_df(attention_result[0][0], 10)
        temp_df(attention_result[0][1], 11)
        sys.exit()
        #==
        result = self.linear_4(attention_result)
        #==
        print(result[0])
        print(result.size())
        temp_df(result[0][0], 10)
        temp_df(result[0][1], 11)
        temp_df(result[1][0], 20)
        temp_df(result[1][1], 21)
        sys.exit()
        #==
        return result
        

class PositionWiseFeedForwadNetwork(nn.Module):
    def __init__(self, size, dropout_p=0.2):
        super().__init__()
        self.Linear_1 = nn.Linear(size, size*4)
        self.Linear_2 = nn.Linear(size*4, size)
        self.Dropout = nn.Dropout(p=dropout_p)
        self.Relu = nn.ReLU()
    
    def forward(self, x):
        z = self.Linear_1(x)
        z_relu = self.Relu(z)
        z_dropout = self.Dropout(z_relu)
        result = self.Linear_2(z_dropout)
        return result
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p):
        super().__init__()
        self.Norm_1 = nn.LayerNorm(d_model)
        self.Norm_2 = nn.LayerNorm(d_model)
        self.Dropout_1 = nn.Dropout(p=dropout_p)
        self.Dropout_2 = nn.Dropout(p=dropout_p)
        self.Multi_Head_Attention = MultiHeadAttention(
            head_num=head_num, 
            d_model=d_model
        )
        self.FFNN = PositionWiseFeedForwadNetwork(
            size=d_model, 
            dropout_p=dropout_p
        )
        
    def forward(self, input, mask):
        # multi head attention
        x_norm = self.Norm_1(input)
        x_m_h_attn = self.Multi_Head_Attention(
            query=x_norm, 
            key=x_norm, 
            value=x_norm, 
            mask=mask
        )
        x_dropout = self.Dropout_1(x_m_h_attn)
        x = input + x_dropout
        
        # feed-forward neural network
        x_norm = self.Norm_2(x)
        x_ffnn = self.FFNN(x_norm)
        x_dropout = self.Dropout_2(x_ffnn)
        x = x + x_dropout
        return x
        
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p):
        super().__init__()
        self.Norm_1 = nn.LayerNorm(d_model)
        self.Norm_2 = nn.LayerNorm(d_model)
        self.Norm_3 = nn.LayerNorm(d_model)
        self.Dropout_1 = nn.Dropout(p=dropout_p)
        self.Dropout_2 = nn.Dropout(p=dropout_p)
        self.Dropout_3 = nn.Dropout(p=dropout_p)
        self.Multi_Head_Attention_1 = MultiHeadAttention(
            head_num=head_num, 
            d_model=d_model
        )
        self.Multi_Head_Attention_2 = MultiHeadAttention(
            head_num=head_num, 
            d_model=d_model
        )
        self.FFNN = PositionWiseFeedForwadNetwork(
            size=d_model, 
            dropout_p=dropout_p
        )
    
    def forward(self, target, encoder_output, target_mask, encoder_mask):
        # masked multi head attention
        x_norm = self.Norm_1(target)
        x_masked_m_h_attn = self.Multi_Head_Attention_1(
            query=x_norm, 
            key=x_norm, 
            value=x_norm, 
            mask=target_mask
        )
        x_dropout = self.Dropout_1(x_masked_m_h_attn)
        x = target + x_dropout
        
        # encoder-decoder attention
        x_norm = self.Norm_2(x)
        enc_output_norm = self.Norm_2(encoder_output)
        x_masked_m_h_attn = self.Multi_Head_Attention_2(
            query=x_norm, 
            key=enc_output_norm, 
            value=enc_output_norm, 
            mask=encoder_mask
        )
        x_dropout = self.Dropout_2(x_masked_m_h_attn)
        x = x + x_dropout
        
        # feed-forward neural network
        x_norm = self.Norm_3(x)
        x_ffnn = self.FFNN(x_norm)
        x_dropout = self.Dropout_3(x_ffnn)
        result = x + x_dropout
        
        return result
        
        
class Embedding(nn.Module):
    def __init__(self, vocab_num, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_num, 
            embedding_dim=d_model
        )
    def forward(self, x):
        x_emb = self.embedding(x)
        result = x_emb * math.sqrt(self.d_model)
        return result
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout_p=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        
        base = torch.ones(d_model // 2).fill_(10000)
        pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model, dtype=torch.float32)
        div_term = torch.pow(base, pow_term)
        
        position = torch.arange(0, max_seq_len).unsqueeze(1)    # 0 부터 순서대로 (max_seq_len, 1)
        pe = torch.zeros(max_seq_len, d_model) # 0으로만 이루어진 (max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        
        # pe를 학습되지 않는 변수로 등록
        self.register_buffer('positional_encoding', pe)
        
    def forward(self, x):
        x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        result = self.dropout(x)

        return result
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)
        
    
    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        result = self.embedding(t)
        return result
    

class Generator(nn.Module):
    def __init__(self, d_model, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model*4)
        self.linear_2 = nn.Linear(d_model*4, output_size)
        
    def forward(self, x):
        x = self.linear_1(x)
        result = self.linear_2(x)
        return result


class Transformer(nn.Module):
    def __init__(self, num_embeddings, d_model, max_seq_len, head_num, dropout_p, layer_num):
        super().__init__()
        self.embedding = Embedding(num_embeddings, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        
        self.encoder = nn.ModuleList([EncoderBlock(d_model, head_num, dropout_p) for _ in range(layer_num)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, head_num, dropout_p) for _ in range(layer_num)])
        
        self.generator = Generator(d_model, num_embeddings)
        
    
    def forward(self, input, target, input_mask, target_mask):
        x_emb = self.embedding(input)
        x_enc = self.positional_encoding(x_emb)
        for encoder_block in self.encoder:
            x_enc = encoder_block(x_enc, input_mask)

        
        tar_emb = self.embedding(target)
        tar_enc = self.positional_encoding(tar_emb)

        for decoder_block in self.decoder:
            tar_enc = decoder_block(tar_enc, x_enc, target_mask, input_mask)
            
        pred = self.generator(tar_enc)

        return pred
    
    def encode(self,input, input_mask):
        x = self.positional_encoding(self.embedding(input))
        for encoder in self.encoder:
            x = encoder(x, input_mask)
        return x

    def decode(self, encode_output, encoder_mask, target, target_mask):
        target = self.positional_encoding(self.embedding(target))
        for decoder in self.decoder:
        #target, encoder_output, target_mask, encoder_mask
            target = decoder(target, encode_output, target_mask, encoder_mask)

        lm_logits = self.generator(target)
        return lm_logits
            
        
        
        
        
        
        