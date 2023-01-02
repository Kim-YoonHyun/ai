'''image'''
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


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
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomRotation(90),
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



'''text'''
import json

import pandas as pd
import numpy as np
import math


# load dataset
def load_raw_data(load_path):
    '''
    csv, xlsx, txt 데이터를 읽어오는 함수.
    column 명은 string, label 로 해야함.

    parameters
    ----------
    load_path: str
        파일의 경로
    
    returns
    -------
    string_list: str list
        문장 데이터를 모아둔 list
    
    label_list: int list    
        라벨 데이터를 모아둔 list
    '''
    import pandas as pd
    if load_path[-3:] == 'csv':
        df = pd.read_csv(load_path, encoding='utf-8')
    if load_path[-4:] == 'xlsx':
        df = pd.read_excel(load_path)
    if load_path[-3:] == 'txt':
        df = pd.read_csv(load_path, sep='\n', engine='python', encoding='utf-8', names=['string'])

    # string, label
    string_list = df['string'].values.tolist()
    try:
        label_list = df['label'].values.tolist()
    except:
        label_list = []
        for _ in string_list:
            label_list.append(0)
    return string_list, label_list


def train_val_test_separate(df, class_list, train_p=0.8, val_p=0.2):
    '''
    문장, label 로 이루어진 df 를 각각 train df, validation df, test df 로 분할 하는 함수

    parameters
    ----------
    df: pandas DataFrame
        string 과 label 로 이루어진 data frame
    
    class_list: str list
        학습에 사용된 class 의 list
    
    train_p: float
        학습, 평가 용으로 나눌때 학습 데이터의 비율
    
    val_p: float
        학습 데이터 중 검증 데이터로 사용할 비율
    
    returns
    -------
    all_train_df: pandas DataFrame
        학습용 string 및 label 을 모아둔 data frame
    all_val_df: pandas DataFrame
        검증용 string 및 label 을 모아둔 data frame
    all_test_df: pandas DataFrame
        평가용 string 및 label 을 모아둔 data frame
    -------

    '''
    train_df_list = []
    val_df_list = []
    test_df_list = []
    
    for class_name in class_list:
        
        # df 에서 특정 클래스 추출
        class_df = df[df['label']==class_name]
        
        # train 비율 계산
        train_num = math.ceil(len(class_df)*train_p)
        val_num = math.ceil(train_num*val_p)
        
        # train, val, test 분할
        train_val_df = class_df.iloc[:train_num, :]
        val_df = train_val_df.iloc[:val_num, :]
        train_df = train_val_df.iloc[val_num:, :]
        test_df = class_df.iloc[train_num:, :]
        
        # list에 추가
        train_df_list.append(train_df)
        val_df_list.append(val_df)
        test_df_list.append(test_df)
    
    # df 화
    all_train_df = pd.concat(train_df_list, axis=0)
    all_val_df = pd.concat(val_df_list, axis=0)
    all_test_df = pd.concat(test_df_list, axis=0)
    
    # 셔플
    all_train_df = all_train_df.sample(frac=1).reset_index(drop=True)
    all_val_df = all_val_df.sample(frac=1).reset_index(drop=True)
    all_test_df = all_test_df.sample(frac=1).reset_index(drop=True)
    
    return all_train_df, all_val_df, all_test_df


def make_json_dataset(load_path, class_dict):
    '''
    raw data를 학습용 dataset 화 시키는 함수

    parameters
    ----------
    load_path: str
        파일의 경로
    
    class_dict: dictionary
        {클래스이름:라벨값, ...} 으로 이루어진 딕셔너리.

    returns
    -------
    new_dataset: json
        미리 구성해놓은 json 데이터로 변환된 dataset
    '''
    # 기반 dict
    new_dataset = {
        'number':{
            'total':[0, {}],
            'train_data':{'total':0},
            'validation_data':{'total':0},
            'test_data':{'total':0}
        },
        'class': class_dict,
        'train_data':{
            'string':None, 
            'label':None
        },
        'validation_data':{
            'string':None, 
            'label':None
        },
        'test_data':{
            'string':None,
            'label':None
            }
    }
    
    # load raw data  f'{args.root_path}/datasets/{args.dataset_name}'
    string_list, label_list = load_raw_data(load_path=load_path)

    # df 화
    df = pd.DataFrame([string_list, label_list], index=['string', 'label']).T
    
    # class 리스트
    class_label_list = list(class_dict.values())
    
    # train, test 로 df 분리
    df_tp = train_val_test_separate(df, class_label_list)
    
    idx = 0
    for tp, tp_str in zip(df_tp, ['train_data', 'validation_data', 'test_data']):
        # 전체 갯수
        new_dataset['number']['total'][0] += len(tp['string'])
        
        for class_label in class_label_list:
            # 첫 루프때 초기값 0 할당
            if idx == 0:
                new_dataset['number']['total'][1][class_label] = 0
            
            # 특정 라벨 값을 가진 df 분리
            df_tp_l = tp[tp['label'] == class_label]
            
            # 전체 기준 각 라벨 갯수 
            new_dataset['number']['total'][1][class_label] += len(df_tp_l)
            
            # train, test 전체 갯수
            new_dataset['number'][tp_str]['total'] += len(df_tp_l)
            
            # train, test 기준 각 라벨 갯수
            new_dataset['number'][tp_str][class_label] = len(df_tp_l)
        
        # train, test 각 데이터 및 라벨 분배
        new_dataset[tp_str]['string'] = tp['string'].values.tolist()
        new_dataset[tp_str]['label'] = tp['label'].values.tolist()
        
        idx += 1
        
    return new_dataset


from torch.utils.data import Dataset
import gluonnlp as nlp
class BERTDataset(Dataset):
    def __init__(self, data_with_label, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.string_list = [i[sent_idx] for i in data_with_label]
        self.string_idx_list = [transform([i[sent_idx]]) for i in data_with_label]
        self.label_list = [np.int32(i[label_idx]) for i in data_with_label]

    def __getitem__(self, i):
        return (self.string_idx_list[i] + (self.label_list[i], ))

    def __len__(self):
        return (len(self.label_list))


def get_string_with_label_data(string_list, label_list):
    '''
    string data에 label 데이터를 덧붙이는 함수

    parameters
    ----------
    string_list: str list
        문장 데이터로 구성된 list
    
    label_list: int list
        각 문장의 라벨값으로 구성된 list
    
    returns
    -------
    (string + label)로 구성된 list data
    
    '''
    string_with_label_data = []
    for string, label in zip(string_list, label_list):
        temp = []
        temp.append(string)
        temp.append(str(label))
        string_with_label_data.append(temp)
    return string_with_label_data


def get_dataset(string_list, label_list, tokenizer, max_len):
    '''
    각 변수를 통해 dataloader 화 시키기 위한 Dataset 클래스를 만드는 함수

    parameters
    ----------
    string_list: str list
        문장 데이터로 구성된 list

    label_list: int list
        각 문장의 라벨값으로 구성된 list

    tokenizer: tokenizer
        토크나이저
    
    max_len: int
        토크나이저 padding 값

    returns
    -------
    dataset: class
        문장 리스트, 토크나이저로 전처리된 string idx 및 각 라벨 값 리스트를 변수로 지니고 있는 클래스 객체.
    '''
    data_with_label = get_string_with_label_data(string_list, label_list)
    dataset = BERTDataset(data_with_label=data_with_label,
                              sent_idx=0,
                              label_idx=1,
                              bert_tokenizer=tokenizer,
                              max_len=max_len,
                              pad=True,
                              pair=False)
    return dataset


def get_dataloader(dataset, batch_size, num_workers):
    '''
    문장 데이터 tokenizing, padding, index 화 등의 전처리 과정을 거친 후
    최종적으로 학습데이터로써 활용할 dataloader 를 얻어내는 함수

    parameters
    ----------
    dataset: class
        string_list, label_list 를 변수로써 지니고 있는 dataset 객체
    
    batch_size: int
        데이터 batch size

    num_workers: int
        DataLoader 변수

    returns
    -------
    dataloader: torch dataloader
        dataloader
    '''
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    return dataloader