'''
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install -U scikit-learn
pip install transformers
'''

import sys
import os

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import pandas as pd
import math
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
from transformers import BertTokenizer
from mylocalmodules import transformer_old as transformer
from mylocalmodules import dataloader_old as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import utils as utm



class TranslationTrainer():
    def __init__(self, dataset, tokenizer, model, max_len, device,
                 model_name, checkpoint_path, batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.ntoken = tokenizer.vocab_size
        self.batch_size = batch_size
    
    
    def my_collate_fn(self, samples):
        b_input_string_list = []
        b_target_string_list = []
        
        input_string_list = []
        input_list = []
        input_mask_list = []
        target_string_list = []
        target_list = []
        target_mask_list = []
        token_num_list = []
        for sample in samples:
            input_string_list.append(sample['input_str'])
            input_list.append(sample['input'])
            input_mask_list.append(sample['input_mask'])
            target_string_list.append(sample['target_str'])
            target_list.append(sample['target'])
            target_mask_list.append(sample['target_mask'])
            token_num_list.append(sample['token_num'])
            
        b_input_string_list.append(input_string_list)
        b_target_string_list.append(target_string_list)
        
        result_dict = {
            'input_str':b_input_string_list,
            'input':torch.stack(input_list).contiguous(),
            'input_mask':torch.stack(input_mask_list).contiguous(),
            'target_str':b_target_string_list,
            'target':torch.stack(target_list).contiguous(),
            'target_mask':torch.stack(target_mask_list).contiguous(),
            'token_num':torch.stack(token_num_list).contiguous()
        }
        return result_dict
        
        
    def build_dataloader(self, train_test_split=0.1, train_shuffle=False, test_shuffle=False):
        dataset_len = len(self.dataset)
        test_len = int(dataset_len * train_test_split)
        train_len = dataset_len - test_len
        # train_dataset, test_dataset = random_split(self.dataset, (train_len, test_len))
        train_dataset = Subset(self.dataset, range(train_len))
        test_dataset = Subset(self.dataset, range(train_len, dataset_len))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            collate_fn=self.my_collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=test_shuffle,
            collate_fn=self.my_collate_fn
        )
        return train_dataloader, test_dataloader
    

    def train(self, epochs, train_dataset, test_dataset, optimizer, scheduler):
        loss = None
        total_loss = 0.0
        global_steps = 0
        loss_dict = {}
        best_val_loss = float('inf')
        best_model = None
        start_epoch = 0
        start_step = 0
        train_dataset_length = len(train_dataset)
        log_interval = 1
        save_interval = 500
        start_time = time.time()
        
        self.model.train()
        self.model.to(self.device)
        # if os.path.isfile()
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            for i, data in enumerate(tqdm(train_dataset)):
                if i < start_step:
                    continue
                input = data['input'].to(self.device)
                input_mask = data['input_mask'].to(self.device)
                target = data['target'].to(self.device)
                target_mask = data['target_mask'].to(self.device)
                
                pred, loss = self.model(
                    input=input, 
                    target=target, 
                    input_mask=input_mask, 
                    target_mask=target_mask, 
                    labels=target
                )
                
                optimizer.zero_grad()
                
                # labels = target
                # if labels is not None:
                #     shift_pred = pred[0][..., :-1, :].contiguous()
                #     shift_pred = shift_pred.view(-1, shift_pred.size(-1))
                    
                #     shift_labels = labels[..., 1:].contiguous()
                #     shift_labels = shift_labels.view(-1)
                    
                #     import torch.nn as nn
                #     loss_function = nn.CrossEntropyLoss(ignore_index=0)
                #     #==
                #     print(shift_pred.size())
                #     print(shift_labels.size())
                #     sys.exit()
                #     #==
                #     loss = loss_function(shift_pred, shift_labels)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                loss_dict[global_steps] = loss.item()
                total_loss += loss.item()
                
                global_steps += 1

                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    total_loss = 0
                    start_time = time.time()
                    
                    if i % save_interval == 0:
                        pass
            val_loss = self.validation(test_dataset)
            self.model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                torch.save({
                    'epoch': epoch,  # 현재 학습 epoch
                    'model_state_dict': best_model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'losses': val_loss,  # Loss 저장
                    'train_step': 1,  # 현재 진행한 학습
                }, f'./{self.model_name}.pth')
                # sys.exit()
                
                
            start_step = 0
            scheduler.step()
    
    
    def validation(self, dataset):
        self.model.eval()
        total_loss = 0.0
        
        self.model.to(self.device)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                input = data['input'].to(self.device)
                input_mask = data['input_mask'].to(self.device)
                target = data['target'].to(self.device)
                target_mask = data['target_mask'].to(self.device)
                
                pred, loss = self.model(
                    input=input,
                    target=target,
                    input_mask=input_mask,
                    target_mask=target_mask,
                    labels=target
                )
                
                # labels = target
                # shift_pred = pred[0][..., :-1, :].contiguous()
                # shift_pred = shift_pred.view(-1, shift_pred.size(-1))
                
                # shift_labels = labels[..., 1:].contiguous()
                # shift_labels = shift_labels.view(-1)
                
                # import torch.nn as nn
                # loss_function = nn.CrossEntropyLoss(ignore_index=0)
                # loss = loss_function(shift_pred, shift_labels)
                total_loss += loss.item()
                
        return total_loss / len(dataset)
        
        


def main():
    torch.manual_seed(42)
    root_path = '/home/kimyh/python/project/transformer'
    data_path = f'{root_path}/data/test_min.csv'
    vocab_path = f'{root_path}/vocab/wiki-vocab.txt'
    checkpoint_path = f'{root_path}/checkpoints'
    
    
    
    model_name = 'transformer-translation-spoken'
    num_embeddings = 25000
    max_length = 15
    d_model = 16
    head_num = 8
    dropout_p = 0.2
    layer_num = 6
    
    
    device_num = 0
    random_seed = 42
    epochs = 200
    batch_size = 2
    learning_rate = 0.5
    
    utm.envs_setting(random_seed)
    device = trm.get_device(device_num)
    
    
    # if torch.cuda.is_available():
    #     device = 'cuda:0'
    # else:
    #     device = 'cpu'
        
    
    
    model = transformer.Transformer(
        num_embeddings=num_embeddings, 
        d_model=d_model, 
        max_seq_len=max_length, 
        head_num=head_num, 
        dropout_p=dropout_p, 
        layer_num=layer_num
    )
    
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    padding_idx = tokenizer.pad_token_id
    

    dataset = dam.TextDataset(
        tokenizer=tokenizer, 
        file_path=data_path,
        max_length=max_length
    )
    
    trainer = TranslationTrainer(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        max_len=max_length,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size
    )
    
    train_dataloader, test_dataloader = trainer.build_dataloader(train_test_split=0.1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer.train(
        epochs=epochs,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    

if __name__  == '__main__':
    main()