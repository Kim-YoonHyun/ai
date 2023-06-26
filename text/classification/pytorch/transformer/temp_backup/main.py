'''
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install -U scikit-learn
pip install transformers
'''

import sys
import os
import argparse
import json

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from transformers import BertTokenizer
from mylocalmodules import transformer
from mylocalmodules import dataloader as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import logutils as lom
from sharemodule import utils as utm


'''
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
                
                pred = self.model(
                    input=input, 
                    target=target, 
                    input_mask=input_mask, 
                    target_mask=target_mask, 
                    labels=target
                )
                
                optimizer.zero_grad()
                
                labels = target
                if labels is not None:
                    shift_pred = pred[0][..., :-1, :].contiguous()
                    shift_pred = shift_pred.view(-1, shift_pred.size(-1))
                    
                    shift_labels = labels[..., 1:].contiguous()
                    shift_labels = shift_labels.view(-1)
                    
                    import torch.nn as nn
                    loss_function = nn.CrossEntropyLoss(ignore_index=0)
                    loss = loss_function(shift_pred, shift_labels)
                    
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
                
                pred = self.model(
                    input=input,
                    target=target,
                    input_mask=input_mask,
                    target_mask=target_mask,
                    labels=target
                )
                
                labels = target
                shift_pred = pred[0][..., :-1, :].contiguous()
                shift_pred = shift_pred.view(-1, shift_pred.size(-1))
                
                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                
                import torch.nn as nn
                loss_function = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_function(shift_pred, shift_labels)
                print(shift_pred)
                print(shift_pred.size())
                print(shift_labels)
                sys.exit()
                total_loss += loss.item()
                
        return total_loss / len(dataset)
        
'''     


# ==
class LossFunction(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function
        
        
    def forward(self, pred, b_label):
        
        # ==
        # shift_pred = pred.contiguous().float()
        # print(shift_pred[0][0])
        # shift_pred = shift_pred.argmax(dim=-1)
        # print(shift_pred[0][0])
        # sys.exit()
        # shift_labels = b_label.contiguous().float()
        # loss = self.loss_function(shift_pred, shift_labels)
        # ==
        
        print(pred.size())
        print(pred[0][0][0])
        shift_pred = pred[..., :-1, :].contiguous()
        print(shift_pred.size())
        print(shift_pred[0][0][0])
        shift_pred = shift_pred.view(-1, shift_pred.size(-1))
        print(shift_pred.size())
        print(shift_pred[0][0])
        sys.exit()
        
        
        shift_labels = b_label[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        
        loss = self.loss_function(shift_pred, shift_labels)
        
        return loss
        


def main():
    # torch.manual_seed(42)
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--dataset_name')
    parser.add_argument('--purpose', type=str)

    # train variable
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--train_p', type=float)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--loss_function_name', type=str, default='CrossEntropyLoss')
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='optimizer learning rate')
    parser.add_argument('--scheduler_name', type=str, default='StepLR')
    parser.add_argument('--gamma', type=float, default=0.95)
    
    # network variable
    parser.add_argument('--max_length', type=int, default=600)
    parser.add_argument('--num_embeddings', type=int, default=10000)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--layer_num', type=int, default=6)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    
    # dataloader variable    
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)

    args = parser.parse_args()
    args_setting = vars(args)
    utm.envs_setting(42)
    
    
    root_path = args.root_path
    dataset_name = args.dataset_name
    device_num = args.device_num
    train_p = args.train_p
    batch_size = args.batch_size
    shuffle = args.shuffle
    drop_last = args.drop_last
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    num_embeddings = args.num_embeddings
    d_model = args.d_model
    max_length = args.max_length
    head_num = args.head_num
    dropout_p = args.dropout_p
    layer_num = args.layer_num
    loss_function_name = args.loss_function_name
    optimizer_name = args.optimizer_name
    learning_rate = args.learning_rate
    scheduler_name = args.scheduler_name
    gamma = args.gamma
    epochs = args.epochs
    max_grad_norm=args.max_grad_norm 
    # =========================================================================
    root_save_path = f'{root_path}/trained_model/{dataset_name}'

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'train.log',
        time_handler=True
    )
    whole_start = time.time()
    # =========================================================================
    # input_list = []
    # label_list = []
    # data_name_list = os.listdir(f'{root_path}/datasets/{dataset_name}')
    # data_name_list.sort()
    # for data_name in data_name_list:
    #     df = pd.read_csv(f'{root_path}/datasets/{dataset_name}/{data_name}', encoding='utf-8-sig')
    #     ECU_VehicleSpeed_STD = [num_embeddings-2] + df['ECU_VehicleSpeed_STD'].values.tolist()
    #     ECU_EngineSpeed = [num_embeddings-2] + df['ECU_EngineSpeed'].values.tolist()
    #     input_list.append(ECU_VehicleSpeed_STD)
    #     label_list.append(ECU_EngineSpeed)
    
    #==
    df = pd.read_csv(f'{root_path}/datasets/{dataset_name}/test_min.csv', encoding='utf-8-sig')
    
    input_list = []
    label_list = []
    kor_list = df['kor'].values.tolist()
    en_list = df['en'].values.tolist()
    
    vocab_path = f'{root_path}/vocab/wiki-vocab.txt'
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    padding_id = tokenizer.pad_token_id
    for kor, en in zip(kor_list, en_list):
        kor_ids_list = tokenizer.encode(kor, max_length=max_length, truncation=True)
        rest = max_length - len(kor_ids_list)
        kor_ids_pad_list = kor_ids_list + [padding_id]*rest
        kor_token_list = tokenizer.convert_ids_to_tokens(kor_ids_pad_list)
        input_list.append(kor_ids_pad_list)
        
        en_ids_list = tokenizer.encode(en, max_length=max_length, truncation=True)
        rest = max_length - len(en_ids_list)
        en_ids_pad_list = en_ids_list + [padding_id]*rest
        en_token_list = tokenizer.convert_ids_to_tokens(en_ids_pad_list)
        label_list.append(en_ids_pad_list)
    #==
    train_num = int(len(input_list) * train_p)
    
    # ===================================================================    
    Train_Dataloader = dam.get_dataloader(
        input_list=input_list[:train_num],
        label_list=label_list[:train_num],
        num_embeddings=num_embeddings,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    Val_Dataloader = dam.get_dataloader(
        input_list=input_list[train_num:],
        label_list=label_list[train_num:],
        num_embeddings=num_embeddings,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # ===================================================================    
    device = trm.get_device(device_num)

    model = transformer.Transformer(
        num_embeddings=num_embeddings, 
        d_model=d_model, 
        max_seq_len=max_length, 
        head_num=head_num, 
        dropout_p=dropout_p, 
        layer_num=layer_num
    )
    model.to(device)
    
    # ===================================================================    
    optimizer = trm.get_optimizer(
        base='torch',
        method=optimizer_name,
        model=model,
        learning_rate=learning_rate
    )
    scheduler = trm.get_scheduler(
        base='torch',
        method=scheduler_name,
        optimizer=optimizer,
        gamma=gamma
    )
    loss_function = trm.get_loss_function(loss_function_name)
    Loss_Function = LossFunction(loss_function)
    
    
    # ============================================================
    condition_order = utm.get_condition_order(
        args_setting=args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose='time_series', 
        start_epoch=0, 
        epochs=epochs, 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=Loss_Function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        reset_class=None, 
        model_save_path=model_save_path
    )
    
    sys.exit()
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