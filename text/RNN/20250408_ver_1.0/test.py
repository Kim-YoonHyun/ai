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
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# import torch.backends.cudnn as cudnn
# import random

# =======================================================
# 변수 지정
# 데이터 변수
max_len = 64  # 문장의 최대 길이 (임의로 설정)

# 네트워크 변수
embed_dim = 16  # 임베딩 차원
hidden_dim = 8 # hidden layer 차원 수
output_dim = 4  # 결과 차원 수 = 라벨 갯수
# 0 : World (세계) 1 : Sports (스포츠) 2 : Business (경제) 3 : Sci/Tec (과학/기술)

# 학습 변수
epochs = 20  # 에포크 값
batch_size = 8  # 데이터 배치 
learning_rate = 0.01   # 학습률

# seed 변수
random_seed = 42  # 랜덤 시드 값
# =======================================================
# torch 랜덤 시드 지정
torch.manual_seed(random_seed)  
# numpy 랜덤 시드 지정
np.random.seed(random_seed)

# =======================================================
with open('./vocab.json', 'r', encoding='utf-8-sig') as f:
    vocab = json.load(f)
with open('./reverse_vocab.json', 'r', encoding='utf-8-sig') as f:
    reverse_vocab = json.load(f)
    
vocab_size = len(vocab)
print(f'vocab size : {vocab_size}')

# =======================================================
# 평가용 데이터 불러오기 (동일)
test_df = pd.read_csv('./data/test_sample_1000.csv', encoding='utf-8-sig')
test_string_ary = test_df['contents'].values
test_label_ary = test_df['label'].values

# =======================================================
# 인코딩 함수
def encode_sentence(sentence, word2idx, max_len=10):
    
    # 문장을 소문자로 변경후 띄어쓰기 단위로 분할
    token_list = sentence.lower().split()
    
    # 인코딩 진행
    encoded = []
    for word in token_list:
            
        # 단어장에 없는 단어는 [UNK] 로 취급        
        try:
            ids = word2idx[word]
        except KeyError:
            word = '[UNK]'
            ids = word2idx[word]
        
        # 인덱스값을 저장
        encoded.append(ids)
    
    
    if len(encoded) > max_len:
        # 문장 길이가 최대길이 (max len) 보다 긴 경우 슬라이싱 진행
        result = encoded[:max_len]
    else:
        # 문장 길이가 최대길이 (max len) 보다 짧은 경우 padding 진행
        result = encoded + [1] * (max_len - len(encoded))
    return result

# =======================================================
# 평가용 데이터에서도 동일하게 진행
test_ids_list = []
for test_string in test_string_ary:
    test_ids = encode_sentence(test_string, vocab, max_len)
    test_ids_list.append(test_ids)
test_ids_ary = np.array(test_ids_list)
    
# # =======================================================
# Dataset 생성 클래스
class CustomDataset(Dataset):
    def __init__(self, x_ary, y_ary):
        self.x_ary = x_ary
        self.y_ary = y_ary
        
    def __len__(self):
        return len(self.x_ary)
        
    def __getitem__(self, idx):
        x = self.x_ary[idx]
        y = self.y_ary[idx]
        return x, y
    
    # 학습용 데이터와 검증용 데이터를 분할하기 위한 인스턴스 함수
    def split_dataset(self, val_ratio=0.2):
        # 데이터의 길이 파악
        data_size = len(self)
        # 검증용 데이터의 비율을 통해 검증용 데이터 갯수 계산
        val_set_size = int(data_size * val_ratio)
        # 통 데이터 갯수에서 검증용 데이터 갯수를 뺀 학습용 데이터 갯수 계산
        train_set_size = data_size - val_set_size
        # random_split 패키지를 통해 데이터 분할
        train_set, val_set = random_split(self, [train_set_size, val_set_size])
        return train_set, val_set

# =======================================================
# Dataset 생성
test_dataset = CustomDataset(test_ids_ary, test_label_ary)
# =======================================================
# Dataloader 패키지를 통해 평가용 dataloader 생성
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# =======================================================
class SimpleNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        # nn.Module 클래스를 상속
        super(SimpleNetwork, self).__init__()
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # RNN 레이어
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=3, batch_first=True)
        # linear 레이어
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    # forward 함수를 통해 Data를 네트워크에 입출력이 가능해짐
    def forward(self, x):
        # 임베딩 진행
        embed = self.embedding(x)
        rnn_out, hidden = self.rnn(embed)
        lin_out = self.linear(rnn_out[:, -1:, :])
        output = lin_out.squeeze()
        return output

# =======================================================
# GPU 또는 CPU 연산장치를 변수에 저장
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model2 = SimpleNetwork(vocab_size, embed_dim, hidden_dim, output_dim)
model2.to(device)

# 학습된 가중치 로딩
weight = torch.load("./trained_model/weight.pt", map_location=device)
model2.load_state_dict(weight)
model2.to(device)

# 모델을 검증 모드로 변경
model2.eval()
# 검증시 가중치 업데이트 등 불필요한 연산 기능 끔
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        output = model2(x)
        predict = torch.argmax(output, dim=-1)
        
        # torch tensor --> numpy array
        x_ary = x.to('cpu').detach().numpy()
        y_ary = y.to('cpu').detach().numpy()
        pred_ary = predict.to('cpu').detach().numpy()
        
        
        
        for ids_ary, label, pred in zip(x_ary, y_ary, pred_ary):
            ids_ary = ids_ary[np.where(ids_ary != 1)]
            
            string = ''
            for ids in ids_ary:
                string = string + ' ' + reverse_vocab[str(ids)]
                
            print(label, pred, string)
        sys.exit()
            
# 0 : World (세계) 1 : Sports (스포츠) 2 : Business (경제) 3 : Sci/Tec (과학/기술)

