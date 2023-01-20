from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 출처: https://deep-deep-deep.tistory.com/34 [딥딥딥:티스토리]
import sys
class TrainDataset(Dataset):
    '''
    데이터를 dataloader 화 시키기 위한 dataset 을 구축하는 class
    '''
    def __init__(self, img_path, annotation_df):
        '''
        parameters
        ----------
        img_path: str
            이미지 파일이 있는 경로

        annotation_df: pandas dataframe
            각 이미지 데이터의 이름 및 라벨값으로 구성된 annotation data.
            annotation 의 column 값은 항상 name, label 이 포함되어야 함.
        '''
        self.img_path = img_path
        self.transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.RandomHorizontalFlip(p=0.4),
            # transforms.RandomVerticalFlip(p=0.4),
            # transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_name_list = annotation_df['name'].values
        self.label_list = annotation_df['label'].values

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        '''
        img shape: (w, h, 3)
        '''
        img = Image.open(f'{self.img_path}/{self.image_name_list[index]}')
        img = self.transforms(img)
        label = int(self.label_list[index])
        
        return img, label


class ValDataset(Dataset):
    def __init__(self, img_path, annotation_df):
        self.img_path = img_path
        self.transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_name_list = annotation_df['name'].values
        self.label_list = annotation_df['label'].values

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        img = Image.open(f'{self.img_path}/{self.image_name_list[index]}')
        img = self.transforms(img)
        label = int(self.label_list[index])
        
        
        return img, label


class TestDataset(Dataset):
    def __init__(self, img_path, annotation_df):
        self.img_path = img_path
        self.transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_name_list = annotation_df['name'].values
    
    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        img = Image.open(f'{self.img_path}/{self.image_name_list[index]}')
        img = self.transforms(img)
        file_name = self.image_name_list[index]
        
        return img, file_name


def get_dataloader(mode, img_path, annotation_df, batch_size, 
            shuffle=False, num_workers=1, pin_memory=True, drop_last=False):
    '''
    dataloader 를 얻어내는 함수

    parameters
    ----------
    mode: str ('train' or 'val' or 'test')
        Dataloader 의 종류를 선택
    
    img_path: str
        이미지 파일이 있는 경로

    annotation_df: pandas df
        데이터로더를 생성하기 위한 annotation 데이터
    
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
    from torch.utils.data import DataLoader

    if mode == 'train':
        dataset = TrainDataset(img_path=img_path, annotation_df=annotation_df)
    if mode == 'val':
        dataset = ValDataset(img_path=img_path, annotation_df=annotation_df)
    if mode == 'test':
        dataset = TestDataset(img_path=img_path, annotation_df=annotation_df)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=drop_last)
    
    return dataloader










