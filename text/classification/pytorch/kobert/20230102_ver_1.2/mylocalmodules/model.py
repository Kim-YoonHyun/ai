import torch
from torch import nn
import numpy as np


def get_bert_model(method, pre_trained, num_labels=None, return_dict=False):
    '''
    bert model 을 구하는 함수

    parameters
    ----------
    pre_trained: str
        모델에 적용할 pre-trained weight

    num_labels: int
        모델이 분류할 라벨 갯수 (BertForSequenceClassification 용)

    return_dict: bool
        dict 를 얻어낼 것인지 여부. (BertModel 용)
    returns
    -------
    model: bert model
        bert 모델
    '''
    if method == 'BertModel':
        from transformers import BertModel
        model = BertModel.from_pretrained(pre_trained, return_dict=False)
    if method == 'BertForSequenceClassification':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(pre_trained, 
                                                              num_labels=num_labels)
    return model
    

    
class BertClassifier(nn.Module):
    def __init__(self,
                 bert,
                 num_classes,
                 hidden_size=768,
                 dr_rate=None,
                 params=None):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

        self.classifier = nn.Linear(hidden_size, num_classes)
    

    def forward(self, string_ids, attention_mask, segment_ids):
        '''
        학습시 활용되는 method
        '''
        _, out = self.bert(input_ids=string_ids, 
                           token_type_ids=segment_ids.long(), 
                           attention_mask=attention_mask.float().to(string_ids.device))
        if self.dr_rate:
            out = self.dropout(out)
        out = self.classifier(out)
        return out