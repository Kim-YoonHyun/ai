import torch
from torch import nn



class Kobert(nn.Module):
    def __init__(self, bert, hidden_size, num_classes, dropout_p=None, params=None):
        super(Kobert, self).__init__()
        self.bert = bert
        self.dropout_p = dropout_p
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dropout_p:
            self.dropout = nn.Dropout(p=dropout_p)
        
        
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(
            input_ids=token_ids, 
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device))
            
        if self.dropout_p:
            out=self.dropout(pooler)
        return self.classifier(out)
    
        
        