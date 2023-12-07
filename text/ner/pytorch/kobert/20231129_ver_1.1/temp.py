import torch
from transformers import BertForTokenClassification, BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# KoBERT 모델 아키텍처 및 토크나이저 불러오기
model = BertForTokenClassification.from_pretrained("monologg/kobert", num_labels=23)  # 6은 클래스의 개수입니다.
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# 데이터셋 로딩 (예시: Hugging Face의 'wikigold' 데이터셋 사용)
dataset = load_dataset("conll2003")
print(next(iter(dataset['train'])))
import sys
sys.exit()

# 토크나이징 및 레이블링 함수 정의
def tokenize_and_label(example):
    tokens = tokenizer(example["tokens"], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    labels = example["ner_tags"]
    return {"input_ids": tokens["input_ids"], "labels": labels}

# 데이터셋을 토크나이징하고 레이블링
tokenized_datasets = dataset.map(tokenize_and_label, batched=True)

# DataLoader 설정
dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# 훈련 루프 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# 모델 저장
model.save_pretrained("path/to/save/ner_model")
