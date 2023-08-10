import sys
import os
import numpy as np 
import gluonnlp as nlp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class InputDataset(Dataset):
    def __init__(self, string_list, label_list, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )
        self.string_list = string_list
        self.token_ids_list = []
        self.valid_length_list = []
        self.segment_ids_list = []
        for string in self.string_list:
            token_ids, valid_length, segment_ids = transform(string)
            self.token_ids_list.append(token_ids)
            self.valid_length_list.append(valid_length)
            self.segment_ids_list.append(segment_ids)
        self.label_list = label_list
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        token_ids = self.token_ids_list[idx]
        valid_length = self.valid_length_list[idx]
        segment_ids = self.segment_ids_list[idx]
        label = self.label_list[idx]
        return token_ids, valid_length, segment_ids, label


def get_dataloader(string_list, label_list, bert_tokenizer, max_len, pad, pair,
                   batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True):
    
    dataset = InputDataset(
        string_list=string_list, 
        label_list=label_list,
        bert_tokenizer=bert_tokenizer,
        max_len=max_len,
        pad=pad,
        pair=pair
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader
    
    

