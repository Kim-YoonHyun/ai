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


def train_val_test_df_separate(df, class_label_list, train_p=0.8, val_p=0.2):
    '''
    문장, label 로 이루어진 df 를 각각 train df, validation df, test df 로 분할 하는 함수

    parameters
    ----------
    df: pandas DataFrame
        string 과 label 로 이루어진 data frame
    
    class_label_list: int list
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
    '''
    import math
    import pandas as pd
    
    train_df_list = []
    val_df_list = []
    test_df_list = []
    
    for class_label in class_label_list:
        # df 에서 특정 클래스 추출
        class_df = df[df['label']==class_label]
        
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
    import pandas as pd
    
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
    
    # train df, val df, test df 로 분리
    train_df, val_df, test_df = train_val_test_df_separate(df, class_label_list, train_p=0.8, val_p=0.2)
    
    idx = 0
    for tvt, tvt_str in zip([train_df, val_df, test_df], ['train_data', 'validation_data', 'test_data']):
        # 전체 갯수 추가
        new_dataset['number']['total'][0] += len(tvt['string'])
        
        # 데이터 갯수 세기
        for class_label in class_label_list:
            # 첫 루프 때 데이터 전체 기준 각 라벨 갯수에 초기값 0 할당
            if idx == 0:
                new_dataset['number']['total'][1][class_label] = 0
            
            # 특정 라벨 만 추출
            df_tvt_l = tvt[tvt['label'] == class_label]
            
            # 데이터 전체 기준 갯수 추가
            new_dataset['number']['total'][1][class_label] += len(df_tvt_l)
            
            # train, val, test 기준 전체 갯수 추가
            new_dataset['number'][tvt_str]['total'] += len(df_tvt_l)
            
            # train, val, test 기준 각 라벨 갯수 추가
            new_dataset['number'][tvt_str][class_label] = len(df_tvt_l)
        
        # 최종 결과 list 화.
        new_dataset[tvt_str]['string'] = tvt['string'].values.tolist()
        new_dataset[tvt_str]['label'] = tvt['label'].values.tolist()
        
        idx += 1
        
    return new_dataset


# bert tokenizer
def get_bert_tokenizer(pre_trained):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(pre_trained, do_lower_case=False)
    return tokenizer

from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
class TextDataset(Dataset):
    def __init__(self, string_list, label_list, bert_tokenizer, max_len):
        '''
        parameters
        ----------
        string_list: str list, shape=(n,)
            문장 리스트
        
        label_list: int list, shape=(n, )
            각 문장별 매겨진 라벨 값 리스트

        bert_tokenizer: tokenizer
            버트 토큰화를 진행할 버트 토크나이저
        
        vocab: vocab
            토큰화를 진행하기 위한 vocab
        
        max_len: int
            문장 토큰화 진행시 적용할 padding 값.
        '''
        self.transform = bert_tokenizer
        self.max_len = max_len
        self.string_list = string_list
        self.label_list = label_list

        string_ids_list, valid_length_list, segment_ids_list = self.get_tokenized_value()
        self.string_ids_list = string_ids_list
        self.attention_mask_list = self.gen_attention_mask(string_ids_list)
        self.segment_ids_list = segment_ids_list


    def get_tokenized_value(self):
        '''
        string 을 tokenize 하는 method

        returns
        -------
        string_ids_list: int list, shape=(문장 갯수, 문장 ids 길이)
            token화 된 문장의 id 값 리스트

        valid_length_list: int list, shape=(문장 갯수, 1)
            문장 id 값 중에 유효한 값(padding 이 아닌 값)

        segment_ids_list: int list, shape=(문장 갯수, 문장 ids 길이)
            문장을 구분하기 위한 id 값.
        '''
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        string_ids_list = []
        valid_length_list = []
        for string in self.string_list:
            new_string = '[CLS] ' + str(string) + ' [SEP]'
            tokenized_string = self.transform.tokenize(new_string)
            string_ids = self.transform.convert_tokens_to_ids(tokenized_string)
            valid_length_list.append(len(string_ids))
            string_ids_list.append(string_ids)
        
        string_ids_list = pad_sequences(string_ids_list, maxlen=self.max_len, dtype='long', truncating='post', padding='post')
        segment_ids_list = np.where(string_ids_list > 0, 0, 0)
        return string_ids_list, valid_length_list, segment_ids_list
    

    def gen_attention_mask(self, string_ids_list):
        '''
        문장의 유효한 id 를 구분하기 위한 mask 리스트를 생성하는 mothod

        parameters
        ----------
        string_ids_list: int list, shape=(문장 갯수, 문장 ids 길이)
            token회 된 문장의 id 값 리스트

        returns
        -------
        attention_mask: int list, shape=(문장 갯수, 문장 ids 길이)
            문장 id 값 중에 유요한 값은 1, 유효하지 않은 값은 0으로 표시한 mask 리스트
        '''
        attention_mask = np.where(string_ids_list > 0, 1, 0)
        return attention_mask


    def __getitem__(self, i):
        return self.string_ids_list[i], self.attention_mask_list[i], self.segment_ids_list[i], self.label_list[i]


    def __len__(self):
        return (len(self.label_list))


def get_dataloader(string_list, label_list, batch_size, tokenizer, max_len, 
                    shuffle, num_workers, pin_memory, drop_last):
    '''
    문장 데이터 tokenizing, padding, index 화 등의 전처리 과정을 거친 후
    최종적으로 학습데이터로써 활용할 dataloader 를 얻어내는 함수

    parameters
    ----------
    string_list: str list, shape=(n, )
        문장 데이터로 구성된 list

    label_list: int list, shape=(n, )
        각 문장의 라벨값으로 구성된 list
    
    batch_size: int
        데이터 batch size
        
    tokenizer: tokenizer
        토크나이저
    
    max_len: int
        토크나이저 padding 값
        
    num_workers: int
        DataLoader 변수

    returns
    -------
    dataloader: torch dataloader
        dataloader
    '''
    from torch.utils.data import DataLoader
    string_list = ['문장1 입니다.', '예시문장 일까요?', '이런 문장도 있지요', '하지만 어쩌면', '이렇게 만들어 볼 수도 있지요']
    label_list = [0, 2, 0, 1, 2]
    
    dataset = TextDataset(string_list=string_list,
                          label_list=label_list,
                          bert_tokenizer=tokenizer, 
                          max_len=max_len)

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=drop_last)
    
    return dataloader
