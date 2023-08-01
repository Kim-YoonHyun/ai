import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class SelfAttention(nn.Module):
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
            # assert w.size() == mask.size()
            mask = mask.unsqueeze(1)
            attention_score = attention_score.masked_fill(mask==False, float('-inf'))
        softmax_attention_score = self.softmax(attention_score)
        result = self.matmul(softmax_attention_score, value)

        return result, softmax_attention_score
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8, d_model=512, dropout_p=0.2):
        super().__init__()
        
        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num
        
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.linear_3 = nn.Linear(d_model, d_model)
        self.linear_4 = nn.Linear(d_model, d_model)
        
        self.self_attention = SelfAttention()
        self.dropout = nn.Dropout(p=dropout_p)
        
    
    def forward(self, query, key, value, mask=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query = self.linear_1(query).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        key = self.linear_2(key).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.linear_3(value).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        
        attention_result, attention_score = self.self_attention(query, key, value, mask)
        attention_result = attention_result.transpose(1, 2).contiguous()
        attention_result = attention_result.view(batch_size, -1, self.head_num * self.d_k)
        result = self.linear_4(attention_result)
        return result
        

class PositionWiseFeedForwadNetwork(nn.Module):
    def __init__(self, d_model, dropout_p=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model*4)
        self.linear_2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        z = self.linear_1(x)
        z_relu = self.relu(z)
        z_dropout = self.dropout(z_relu)
        result = self.linear_2(z_dropout)
        return result
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.Multi_Head_Attention = MultiHeadAttention(head_num, d_model, dropout_p)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        
        self.norm_2 = nn.LayerNorm(d_model)
        self.FFNN = PositionWiseFeedForwadNetwork(d_model, dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)

        
    def forward(self, input, mask):
        # multi head attention layer
        x_norm = self.norm_1(input)
        x_m_h_attn = self.Multi_Head_Attention(x_norm, x_norm, x_norm, mask)
        x_dropout = self.dropout_1(x_m_h_attn)
        x = input + x_dropout
        
        # feed-forward layer
        x_norm = self.norm_2(x)
        x_ffnn = self.FFNN(x_norm)
        x_dropout = self.dropout_2(x_ffnn)
        x = x + x_dropout
        return x
        
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, head_num, dropout_p):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.masked_Multi_Head_Attention = MultiHeadAttention(head_num, d_model, dropout_p)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        
        self.norm_2 = nn.LayerNorm(d_model)
        self.encoder_decoder_attention = MultiHeadAttention(head_num, d_model, dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)
        
        self.norm_3 = nn.LayerNorm(d_model)
        self.FFNN = PositionWiseFeedForwadNetwork(d_model, dropout_p)
        self.dropout_3 = nn.Dropout(p=dropout_p)
        
    
    def forward(self, target, encoder_output, target_mask, encoder_mask):
        x_norm = self.norm_1(target)
        x_masked_m_h_attn = self.masked_Multi_Head_Attention(x_norm, x_norm, x_norm, target_mask)
        x_dropout = self.dropout_1(x_masked_m_h_attn)
        x = target + x_dropout
        
        x_norm = self.norm_2(x)
        enc_output_norm = self.norm_2(encoder_output)
        x_masked_m_h_attn = self.encoder_decoder_attention(x_norm, enc_output_norm, enc_output_norm, encoder_mask)
        x_dropout = self.dropout_2(x_masked_m_h_attn)
        x = x + x_dropout
        
        
        x_norm = self.norm_3(x)
        x_ffnn = self.FFNN(x_norm)
        x_dropout = self.dropout_3(x_ffnn)
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
        
    
    def forward(self, input, target, input_mask, target_mask, labels):
        x_emb = self.embedding(input)
        x_enc = self.positional_encoding(x_emb)
        for encoder_block in self.encoder:
            x_enc = encoder_block(x_enc, input_mask)

        
        tar_emb = self.embedding(target)
        tar_enc = self.positional_encoding(tar_emb)

        for decoder_block in self.decoder:
            tar_enc = decoder_block(tar_enc, x_enc, target_mask, input_mask)
            
        pred = self.generator(tar_enc)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = pred[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            print(shift_labels.size())
            print(shift_logits.view(-1, shift_logits.size(-1)).size())
            print(shift_labels.view(-1).size())
            sys.exit()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return pred, loss
    
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
            
        
        
        
        
        
        