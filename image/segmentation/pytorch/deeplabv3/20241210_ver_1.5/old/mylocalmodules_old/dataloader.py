from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys


class ImageDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([224, 224]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_list, self.label_list = self.get_img_label_list(
            img_path=img_path,
            label_path=label_path
        )


    def get_img_label_list(self, img_path, label_path):
        img_list = []
        img_name_list = np.sort(os.listdir(img_path)).tolist()
        for img_name in tqdm(img_name_list):
            img = Image.open(f'{img_path}/{img_name}')
            img = np.array(img)[:, :, :3]
            img = self.transforms(img)
            img_list.append(img)
        
        label_list = []
        if label_path:
            label_name_list = np.sort(os.listdir(label_path)).tolist()
            for label_name in tqdm(label_name_list):
                label = Image.open(f'{label_path}/{label_name}')
                label = self.transforms(label)
                label = torch.squeeze(label)
                label = (label*255).int()
                label_list.append(label)
        else:
            label_list = np.full(len(img_list), 0)
        self.img_name_list = img_name_list
        return img_list, label_list


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        return self.img_list[idx], self.label_list[idx]


def get_dataloader(img_path, label_path, batch_size, 
            shuffle=False, num_workers=1, pin_memory=True, drop_last=False):
    '''
    dataloader 를 얻어내는 함수

    parameters
    ----------
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
    dataset = ImageDataset(
        img_path=img_path, 
        label_path=label_path
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    return dataloader