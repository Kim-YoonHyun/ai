import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys


class TrainDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        self.img_path = img_path
        self.label_path = label_path
        self.transforms = transforms.Compose([
            # transforms.Resize([224, 224]),
            # transforms.RandomHorizontalFlip(p=0.4),
            # transforms.RandomVerticalFlip(p=0.4),
            # transforms.RandomRotation(90),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_name_list = np.sort(os.listdir(img_path)).tolist()
        self.label_name_list = np.sort(os.listdir(label_path)).tolist()


    def __len__(self):
        return len(self.img_name_list)


    def __getitem__(self, index):
        img = Image.open(f'{self.img_path}/{self.img_name_list[index]}')
        img = np.array(img)[:, :, :3]
        img = self.transforms(img)
        
        label = Image.open(f'{self.label_path}/{self.label_name_list[index]}')
        label = self.transforms(label)
        label = torch.squeeze(label)
        label = (label*255).int()

        return img, label


class ValDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.transforms = transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_name_list = np.sort(os.listdir(img_path)).tolist()
        self.label_name_list = np.sort(os.listdir(label_path)).tolist()
        

    def __len__(self):
        return len(self.img_name_list)


    def __getitem__(self, index):
        img = Image.open(f'{self.img_path}/{self.img_name_list[index]}')
        img = np.array(img)[:, :, :3]
        img = self.transforms(img)
        
        label = Image.open(f'{self.label_path}/{self.label_name_list[index]}')
        label = self.transforms(label)
        label = torch.squeeze(label)
        label = (label*255).int()

        return img, label


class TestDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        self.img_path = img_path
        self.label_path = label_path
        self.transforms = transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_name_list = np.sort(os.listdir(img_path)).tolist()
        self.label_name_list = np.sort(os.listdir(label_path)).tolist()


    def __len__(self):
        return len(self.img_name_list)


    def __getitem__(self, index):
        img = Image.open(f'{self.img_path}/{self.img_name_list[index]}')
        img = np.array(img)[:, :, :3]
        img = self.transforms(img)
        
        label = Image.open(f'{self.label_path}/{self.label_name_list[index]}')
        label = self.transforms(label)
        label = torch.squeeze(label)
        label = (label*255).int()

        return img, label


def get_dataloader(mode, img_path, label_path, batch_size, 
            shuffle=False, num_workers=1, pin_memory=True, drop_last=False):
    '''
    dataloader 를 얻어내는 함수

    parameters
    ----------
    mode: str ('train' or 'val' or 'test')
        Dataloader 의 종류를 선택
    
    img_path: str
        이미지 파일이 있는 경로

    label_path: str
        label 파일이 있는 경로
    
    batch_size: int
        데이터 배치 사이즈
    
    shuffle: bool
        데이터 셔플 여부

    num_workers: int
        데이터 로딩 시 활용할 subprocessor 갯수
    
    pin_memory: bool
        cuda 고정 메모리 사용 여부
    
    drop_last: bool
        마지막 batch 사용 여부

    returns
    -------
    dataloader: dataloader
        데이터 배치가 적용된 데이터 로더
    '''
    if mode == 'train':
        dataset = TrainDataset(img_path=img_path, label_path=label_path)
    if mode == 'val':
        dataset = ValDataset(img_path=img_path, label_path=label_path)
    if mode == 'test':
        dataset = TestDataset(img_path=img_path, label_path=label_path)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    return dataloader