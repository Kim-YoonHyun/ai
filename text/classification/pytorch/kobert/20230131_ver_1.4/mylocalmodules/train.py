import numpy as np
import sys
import copy
import os
import time
import pandas as pd
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch import nn

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils
from sharemodule import trainutils as tutils


class Iterator():
    def __init__(self, dataloader, model, device):
        batch_idx_list = []
        b_string_ids_list = []
        b_attention_mask_list = []
        b_segment_ids_list = []
        b_label_list = []
        
        for batch_idx, (b_string_ids, b_attention_mask, b_segment_ids, b_label) in enumerate(dataloader):
            batch_idx_list.append(batch_idx)
            b_string_ids_list.append(b_string_ids)
            b_attention_mask_list.append(b_attention_mask)
            b_segment_ids_list.append(b_segment_ids)
            b_label_list.append(b_label)
            
        self.model = model
        self.device = device
        self.batch_idx_list = batch_idx_list
        self.b_string_ids_list = b_string_ids_list
        self.b_attention_mask_list = b_attention_mask_list
        self.b_segment_ids_list = b_segment_ids_list
        self.b_label_list = b_label_list
        self.count = -1
    
    def __len__(self):
        return len(self.batch_idx_list)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.batch_idx_list) - 1:
            self.count += 1
            
            b_string_ids = self.b_string_ids_list[self.count]
            b_attention_mask = self.b_attention_mask_list[self.count]
            b_segment_ids = self.b_segment_ids_list[self.count]
            b_label = self.b_label_list[self.count]
            
            x = b_string_ids.long().to(self.device)
            b_attention_mask = b_attention_mask.long().to(self.device)
            b_segment_ids = b_segment_ids.long().to(self.device)
            b_label = b_label.long().to(self.device)
            
            pred = self.model(
                x,
                token_type_ids=None, 
                attention_mask=b_attention_mask, 
                labels=b_label
            )
            pred = pred[1]

            return pred, b_label
        else:
            raise StopIteration


