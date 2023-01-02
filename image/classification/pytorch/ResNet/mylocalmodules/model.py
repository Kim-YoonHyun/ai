'''image'''
import timm
import timm2
from torch import nn


class EffNet(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(EffNet, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features = 625, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_outputs)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output


class Swin_v2_cr_huge_224(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(Swin_v2_cr_huge_224, self).__init__()
   
        self.model = timm2.models.create_model('swinv2_cr_huge_224', pretrained=True)
        self.model.head = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_outputs)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output


class Swin_v2_cr_small_224(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(Swin_v2_cr_small_224, self).__init__()
   
        self.model = timm2.models.create_model('swinv2_cr_small_224', pretrained=True)
        self.model.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_outputs)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output

# F = nn.functional

class CustomNet(nn.Module):

    def __init__(self, n_inputs:int, n_outputs:int, **kwargs):
        """
        Args:
            n_input(int): feature 수
            n_output(int): class 수

        Notes:
            fc: fully connected layer
        """
        super(CustomNet, self).__init__()
        self.n_input = n_inputs
        self.n_output = n_outputs

        self.linear = nn.Linear(self.n_input, self.n_output)

    def forward(self, x):
        output = self.linear(x)
        
        return output


def get_model(model_name:str, model_args:dict):
    if model_name == 'Linear':
        return CustomNet(**model_args)
    if model_name == 'effnet':
        return EffNet(**model_args)
    if model_name == 'swin_v2_cr_small_224':
        return Swin_v2_cr_small_224(**model_args)
    if model_name == 'swin_v2_cr_huge_224':
        return Swin_v2_cr_huge_224(**model_args)


'''text'''
import torch
from torch import nn
import numpy as np
    
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 num_classes,
                 hidden_size=768,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids=token_ids, 
                              token_type_ids=segment_ids.long(), 
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)