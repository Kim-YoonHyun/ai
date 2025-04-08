'''
conda install pytorch torchtext torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas
pip install nltk
'''

import sys
import os
# import time
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# import torch.backends.cudnn as cudnn
# import random
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
# cudnn.benchmark = False
# cudnn.deterministic = True

# random.seed(random_seed)


# from tqdm.notebook import tqdm_notebook

# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator


CUDA_LAUNCH_BLOCKING=1
        
# =======================================================
df = pd.read_csv('./data/train_sample.csv', encoding='utf-8-sig')
string_ary = df['contents'].values
label_ary = df['label'].values


test_df = pd.read_csv('./data/test_sample.csv', encoding='utf-8-sig')
test_string_ary = test_df['contents'].values
test_label_ary = test_df['label'].values

10321
# =======================================================
# 2. 간단한 토큰화 및 단어 집합 만들기
vocab_set = set([])
for string in string_ary:
    string_split = string.lower().split()
    for word in string_split:
        vocab_set = vocab_set.union(set([word]))
vocab_set = sorted(vocab_set)
        
word2idx = {'[UNK]':0, '[PAD]':1}
for idx, word in enumerate(vocab_set):
    word2idx[word] = idx+2
vocab_size = len(word2idx)  # 패딩 토큰을 위한 공간 추가

# =======================================================
# 텍스트를 인덱스 값으로 변환하는 함수
def encode_sentence(sentence, word2idx, max_len=10):
    token_list = sentence.lower().split()
    
    encoded = []
    for word in token_list:
            
        # 단어장에 없는 단어는 [UNK] 로 취급        
        try:
            ids = word2idx[word]
        except KeyError:
            word = '[UNK]'
            ids = word2idx[word]
            
        encoded.append(ids)
        
    if len(encoded) > max_len:
        result = encoded[:max_len]
    else:
        # padding
        result = encoded + [1] * (max_len - len(encoded))
    return result

# =======================================================
# 텍스트를 인덱스 값으로 변환
max_len = 128  # 문장의 최대 길이 (임의로 설정)
print(f'max len : {max_len}')
ids_list = []
for string in string_ary:
    ids = encode_sentence(string, word2idx, max_len)
    ids_list.append(ids)
ids_ary = np.array(ids_list)

test_ids_list = []
for test_string in test_string_ary:
    test_ids = encode_sentence(test_string, word2idx, max_len)
    test_ids_list.append(test_ids)
test_ids_ary = np.array(test_ids_list)
    
# =======================================================
# Dataset 생성 클래스
class CustomDataset(Dataset):
    def __init__(self, x_ary, y_ary):
        self.x_ary = x_ary
        self.y_ary = y_ary
        
    def __len__(self):
        return len(self.x_ary)
        
    def __getitem__(self, idx):
        x = self.x_ary[idx]
        # x = torch.tensor(x)
        y = self.y_ary[idx]
        # y = torch.tensor(y)
            
        return x, y
    
    def split_dataset(self, val_ratio=0.2):
        data_size = len(self)
        val_set_size = int(data_size * val_ratio)
        train_set_size = data_size - val_set_size
        
        train_set, val_set = random_split(self, [train_set_size, val_set_size])
        return train_set, val_set

# =======================================================
# Dataset 생성
dataset = CustomDataset(ids_ary, label_ary)
train_dataset, val_dataset = dataset.split_dataset(0.2)
test_dataset = CustomDataset(test_ids_ary, test_label_ary)


# =======================================================
# dataloader 생성
batch_size = 8
print(f'batch size : {batch_size}')

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_dataloader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# =======================================================
# simple ai network 구축
vocab_size = len(vocab_set)
embed_dim = 32
hidden_dim = 16
output_dim = 4

print(f'embed_dim: {embed_dim} --> hidden_dim: {hidden_dim}')
print(f'output_dim: {output_dim}')
print(f'layer num : 3')

class SimpleNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embed = self.embedding(x)
        rnn_out, hidden = self.rnn(embed)
        lin_out = self.linear(rnn_out[:, -1:, :])
        output = lin_out.squeeze()
        return output

# =======================================================
# device, model 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model = SimpleNetwork(vocab_size, embed_dim, hidden_dim, output_dim)
model.to(device)

# =======================================================
# criterion, opimizer, scheduler 생성
learning_rate = 0.001
epochs = 20

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

# =======================================================
# 학습 함수
# def train(dataloader, epoch):
#     model.train()
#     train_acc = 0
#     train_count = 0
    
#     log_interval = 2000
    
#     for idx, (text, label) in enumerate(dataloader):
#         text = torch.tensor(text, dtype=torch.float)
#         text = text.to(device)
#         label = torch.tensor(label, dtype=torch.float)
#         label = label.to(device)
        
#         optimizer.zero_grad()
        
#         output = model(text)
        
#         predict = torch.argmax(output, dim=-1)
#         print(type(output))
#         print(type(label))
#         sys.exit()
#         loss = criterion(output, label)
#         loss.backward()
        
#         optimizer.step()
        
#         train_acc += (predict == label).sum().item()
        
#         train_count += label.size(0)
        
#         if idx % log_interval == 0 and idx > 0:
#              print(f'| epoch {epoch} | {idx}/{len(dataloader)} batches | accuracy {train_acc / train_count}')
        
#     scheduler.step()    
    
# =======================================================
# 평가 함수
def evaluate(dataloader):
    model.eval()
    val_acc = 0
    val_count = 0
    val_acc_item_list = []

    with torch.no_grad():
        for idx, (text, label, offset) in enumerate(dataloader):
            result = model(text, offset)

            predict = torch.argmax(result, dim=-1)

            acc_item = (label == predict).sum().item()
            val_acc_item_list.append(acc_item)

            val_count += label.size(0)
            val_acc = np.sum(val_acc_item_list) / val_count
    return val_acc


# =======================================================
# 학습 진행
total_acc = 0
for epoch in range(epochs):
    model.train()
    train_acc = 0
    train_loss = 0
    log_interval = 10
    
    for x, y in train_dataloader:
        print(epoch, x.size())
        x = x.to(device)
        y = y.to(device)#, dtype=torch.float)
        
        # x_input = x.to(device, dtype=torch.float)
        # y_input = y.to(device, dtype=torch.float)
        # x_mark = torch.tensor(x_mark, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        output = model(x)
        
        # print(output.size())
        # print(y.size())
        # print('-----------------------')
        
        #==
        predict = torch.argmax(output, dim=-1)
        # print(predict)
        #==
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
        
    # if epoch % log_interval == 0:
    #     print(f'| epoch {epoch} / {epochs} | accuracy {train_acc / len(train_dataloader)}')
        
    scheduler.step()   
    sys.exit()
    
    
    
    acc_val = evaluate(val_dataloader)
    
    if total_acc < acc_val:
        total_acc = acc_val
        
    print('-' * 60)
    print(f'| end of epoch {epoch} | valid accuracy {total_acc}')
    print('-' * 60)
    
sys.exit()

acc_val = evaluate(test_dataloader)
print('-' * 59)
print(f'test accuracy {acc_val}')
print('-' * 59)

