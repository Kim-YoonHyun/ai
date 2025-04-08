'''
conda install pytorch torchtext torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas
pip install nltk
'''

import sys
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# =======================================================
# 데이터 변수
max_len = 64  # 문장의 최대 길이 (임의로 설정)

# 네트워크 변수
embed_dim = 16  # 임베딩 차원
hidden_dim = 8 # hidden layer 차원 수
output_dim = 4  # 결과 차원 수 = 라벨 갯수
# 0 : World (세계) 1 : Sports (스포츠) 2 : Business (경제) 3 : Sci/Tec (과학/기술)

# 학습 변수
epochs = 100  # 에포크 값
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
# 학습용 데이터 불러오기
# pandas 패키지를 이용해서 csv 파일을 utf-8 엔코딩 형식으로 불러오기
df = pd.read_csv('./data/train_sample_3000.csv', encoding='utf-8-sig') 
# pandas dataframe 의 'contents' 열을 numpy array 로 추출
string_ary = df['contents'].values   
# pandas dataframe 의 'label' 열을 numpy array 로 추출
label_ary = df['label'].values  

# =======================================================
# 이미 만들어둔 vocab 파일이 존재하는 경우
if os.path.isfile('./vocab.json'):
    # vocab 불러오기
    with open('./vocab.json', 'r', encoding='utf-8-sig') as f:
        vocab = json.load(f)
    # reverse vocab 불러오기
    with open('./reverse_vocab.json', 'r', encoding='utf-8-sig') as f:
        reverse_vocab = json.load(f)
# 만들어둔 vocab 파일이 존재하지 않는 경우
else:
    # 빈 단어장 생성
    word_set = set([])
    # 학습용 데이터를 활용해 단어장 생성
    # tqdm: 진행 상태 bar 를 보여주는 패키지
    for string in tqdm(string_ary):
        # 소문자로 변경후 띄어쓰기 단위로 분할
        string_split = string.lower().split()
        # 분할된 단어를 빈 단어장에 추가
        for word in string_split:
            word_set = word_set.union(set([word]))
    # 단어장 집합(set)을 정렬
    word_set = sorted(word_set)

    # 미식별 토큰 UNK 와 패팅 토큰 PAD 를 가지고 있는 vocab 변수
    vocab = {'[UNK]':0, '[PAD]':1}
    reverse_vocab = {0:'[UNK]', 1:'[PAD]'}
    # 단어장에 있는 단어들에 대해 순서대로 id 값 지정
    # enumerate : 순서 index 값과 변수내 요소값을 같이 출력하는 방법
    for idx, word in enumerate(word_set):
        vocab[word] = idx+2
        reverse_vocab[idx+2] = word
        
    # 완성한 vocab 저장
    with open('./vocab.json', 'w', encoding='utf-8-sig') as f:
        json.dump(vocab, f, indent='\t', ensure_ascii=False)
    # 완성한 reverse vocab 저장
    with open('./reverse_vocab.json', 'w', encoding='utf-8-sig') as f:
        json.dump(reverse_vocab, f, indent='\t', ensure_ascii=False)
    
# vocab 의 길이 설정
vocab_size = len(vocab)
print(f'vocab size : {vocab_size}')

# =======================================================
# 인코딩 함수
def encode_sentence(sentence, vocab, max_len=10):
    
    # 문장을 소문자로 변경후 띄어쓰기 단위로 분할
    token_list = sentence.lower().split()
    
    # 인코딩 진행
    encoded = []
    for word in token_list:
            
        # 단어장에 없는 단어는 [UNK] 로 취급        
        try:
            ids = vocab[word]
        except KeyError:
            word = '[UNK]'
            ids = vocab[word]
        
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
print(f'max len : {max_len}')
# 텍스트를 인덱스 값으로 변환
ids_list = []
# 학습 데이터 별로 진행
for string in string_ary:
    # 인코딩 함수를 통해 문장을 인덱스화
    ids = encode_sentence(string, vocab, max_len)
    ids_list.append(ids)
# list 를 numpy array 로 변경 (향후 학습시 GPU 연산을 위함)
ids_ary = np.array(ids_list)

# =======================================================
# Dataset 생성 클래스
class CustomDataset(Dataset):
    def __init__(self, x_ary, y_ary):
        self.x_ary = x_ary
        self.y_ary = y_ary
        
    def __len__(self):
        return len(self.x_ary)
    
    # dataset 에서 idx 번째의 값을 추출할 수 있도록 함
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
dataset = CustomDataset(ids_ary, label_ary)
# 인스턴스 함수를 통해 train(학습), val (검증:validation) dataset 분할
train_dataset, val_dataset = dataset.split_dataset(0.2)

# =======================================================
print(f'batch size : {batch_size}')
# Dataloader 패키지를 통해 학습용 dataloader 생성
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)
# Dataloader 패키지를 통해 검증용 dataloader 생성
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# =======================================================
# RNN 을 활용한 간단한 ai network 구축
print(f'embed_dim: {embed_dim}')
print(f'hidden_dim: {hidden_dim}')
print(f'output_dim: {output_dim}')
print(f'layer num : 3')

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
# 만들어둔 Network 클래스를 통해 model 을 생성
model = SimpleNetwork(vocab_size, embed_dim, hidden_dim, output_dim)
# model 을 연산장치에 올림
model.to(device)

# =======================================================
# 손실함수 생성
criterion = torch.nn.CrossEntropyLoss()
# 옵티마이저 생성
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 학습률 조정용 스케줄러 생성
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
    
# =======================================================
# 학습 진행
# 몇번째 에포크마다 검증을 진행할것인지 정하는 검증 간격 변수
# 에포크 4번째 마다 진행
val_interval = 4
# 최적의 검증 loss 값을 무한대 값으로 초기화
best_loss = float('inf')
for epoch in range(epochs):
    
    # 모델을 학습 모드로 변경
    model.train()
    
    # 학습 loss 초기화
    train_loss = 0
    
    # dataloader 에서 값을 뽑아서 진행
    for x, y in train_dataloader:
        # x 및 y 값을 연산장치에 올림
        x = x.to(device)
        y = y.to(device)
            
        # optimizer 의 파라미터를 모두 0 으로 초기화
        optimizer.zero_grad()
        
        # 제작해둔 AI Network에 데이터를 입력하여 결과값 얻어냄
        output = model(x)
        
        # AI 예측 결과 라벨을 얻기위해 예측 결과 output 에 대하여 argmax 진행
        # argmax: 가장 값이 큰 요소의 index 번호를 얻어냄
        predict = torch.argmax(output, dim=-1)
        
        # 손실함수를 통해 결과값 output 과 정답 y 간의 loss 를 계산
        loss = criterion(output, y)
        # loss 를 back propagation 을 진행하여 가중치 계산
        loss.backward()
        # 계산된 가중치 파라미터를 업데이트
        optimizer.step()
        
        # 한 epoch 내의 모든 loss 값 더하기
        train_loss += loss
    train_loss = train_loss / len(train_dataloader)
    
    # 해당 에포크에서의 loss 값 출력        
    print(f'| epoch {epoch+1} / {epochs} | train loss {train_loss}')
    
    # 학습률 조정
    scheduler.step()   
    
    # 첫 에포크를 제외한 4번째 에포크마다 검증 진행
    if (epoch + 1) % val_interval == 0 and epoch > 0:
        # 모델을 검증 모드로 변경
        model.eval()
        val_loss = 0
        # 검증시 가중치 업데이트 등 불필요한 연산 기능 끔
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                # predict = torch.argmax(output, dim=-1)
                
                # 손실함수를 통해 결과값 output 과 정답 y 간의 loss 를 계산
                loss = criterion(output, y)
                val_loss += loss
                
            val_loss = val_loss / len(val_dataloader)

        print('-' * 60)
        print(f'| epoch {epoch+1} / {epochs} | val loss {val_loss}')
        print('-' * 60)
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'./trained_model/weight.pt')
            
print(f'best epoch: {epoch}')

