import torch
import torch.nn as nn


# class TokenEmbedding(nn.Module):
#     def __init__(self, embed_size, d_model):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(embed_size, d_model)

#     def forward(self, x):
#         x = self.embedding(x)
#         return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.require_grad = False

        # position
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # div term
        base = torch.tensor(10000).float()
        term = torch.arange(0, d_model, 2).float()
        div_term = base.pow((term / d_model))
        
        # positional embedding
        pos_emb[:, 0::2] = torch.sin(position / div_term)
        pos_emb[:, 1::2] = torch.cos(position / div_term)
        pos_emb = pos_emb.unsqueeze(0)
        
        # 학습되지 않는 파라미터 등록
        self.register_buffer('pos_emb', pos_emb)
        
    def forward(self):
        return self.pos_emb
    

class DataEmbedding(nn.Module):
    def __init__(self, max_len, d_model, vocab_size, dropout_p=0.1):
        super(DataEmbedding, self).__init__()
        
        # token embedding        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # positional embeding
        self.position_embedding = PositionalEmbedding(
            max_len=max_len,
            d_model=d_model, 
        )
        # dropout            
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding()[:, :x.size(1)]
        x = self.dropout(x)
        return x