import sys
import pandas as pd

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils


# # load dataset
# def load_raw_data(load_path):
#     '''
#     csv, xlsx, txt 데이터를 읽어오는 함수.
#     column 명은 string, label 로 해야함.

#     parameters
#     ----------
#     load_path: str
#         파일의 경로
    
#     returns
#     -------
#     string_list: str list
#         문장 데이터를 모아둔 list
    
#     label_list: int list    
#         라벨 데이터를 모아둔 list
#     '''
#     import pandas as pd
#     if load_path[-3:] == 'csv':
#         df = pd.read_csv(load_path, encoding='utf-8')
#     if load_path[-4:] == 'xlsx':
#         df = pd.read_excel(load_path)
#     if load_path[-3:] == 'txt':
#         df = pd.read_csv(load_path, sep='\n', engine='python', encoding='utf-8', names=['string'])

#     # string, label
#     string_list = df['string'].values.tolist()
#     try:
#         string_class_list = df['class'].values.tolist()
#     except:
#         string_class_list = np.full(len(df), 0).tolist()

#     return string_list, string_class_list


# def train_val_test_df_separate(df, class_list, train_p=0.8):
#     '''
#     문장, label 로 이루어진 df 를 각각 train df, validation df, test df 로 분할 하는 함수

#     parameters
#     ----------
#     df: pandas DataFrame
#         string 과 label 로 이루어진 data frame
    
#     class_label_list: int list
#         학습에 사용된 class 의 list
    
#     train_p: float
#         학습용으로 나눌때 학습 데이터의 비율
    
#     returns
#     -------
#     all_train_df: pandas DataFrame
#         학습용 string 및 label 을 모아둔 data frame
#     all_val_df: pandas DataFrame
#         검증용 string 및 label 을 모아둔 data frame

#     '''
#     import math
#     import pandas as pd

#     train_df_list = []
#     val_df_list = []
#     for class_name in class_list:
#         # df 에서 특정 클래스 추출
#         class_df = df[df['class']==class_name]
        
#         # train 비율 계산
#         # train_num = math.ceil(len(class_df)*train_p)
#         train_num = int(len(class_df)*train_p)
        
#         # train, val 분할
#         train_df = class_df.iloc[:train_num, :]
#         val_df = class_df.iloc[train_num:, :]
        
#         # list에 추가
#         train_df_list.append(train_df)
#         val_df_list.append(val_df)
    
#     # df 화
#     all_train_df = pd.concat(train_df_list, axis=0)
#     all_val_df = pd.concat(val_df_list, axis=0)
    
#     # 셔플
#     all_train_df = all_train_df.sample(frac=1).reset_index(drop=True)
#     all_val_df = all_val_df.sample(frac=1).reset_index(drop=True)

#     return all_train_df, all_val_df


# def make_json_dataset(load_path, train_p=None):
#     dataset_info = {
#         'name':None,
#         'number':{
#             'total':[0, {}],
#             'train_data':{'total':0},
#             'validation_data':{'total':0},
#         },
#         'class_dict': None,
#     }

#     train_data = {
#         'string':None, 
#         'label':None
#     }

#     val_data = {
#         'string':None, 
#         'label':None
#     }
    
#     string_list, string_class_list = load_raw_data(load_path=load_path)


#     uni_class_list = np.unique(string_class_list).tolist()
#     class_dict = utils.make_class_dict(uni_class_list)
#     uni_label_list = list(class_dict.values())

#     dataset_info['name'] = load_path
#     dataset_info['class_dict'] = class_dict

#     string_class_ary = np.array(string_class_list.copy())
#     for uni_class, uni_label in zip(uni_class_list, uni_label_list):
#         string_class_ary = np.where(string_class_ary==uni_class, uni_label, string_class_ary)
#     string_label_list = list(map(int, string_class_ary))

#     # df 화
#     df = pd.DataFrame([string_list, string_class_list, string_label_list], index=['string', 'class', 'label']).T
    
#     # train df, val df 로 분리
#     train_df, val_df = train_val_test_df_separate(df, uni_class_list, train_p=train_p)

#     idx = 0

#     for tv_df, tv, tv_json in zip([train_df, val_df], ['train_data', 'validation_data'], [train_data, val_data]):
#         if not tv_df.empty:
#             dataset_info['number']['total'][0] += len(tv_df['string'])
#             for uni_class_name in uni_class_list:
#                 if idx == 0:
#                     dataset_info['number']['total'][1][uni_class_name] = 0
#                 tv_df_class = tv_df[tv_df['class'] == uni_class_name]
#                 dataset_info['number']['total'][1][uni_class_name] += len(tv_df_class)
#                 dataset_info['number'][tv]['total'] += len(tv_df_class)
#                 dataset_info['number'][tv][uni_class_name] = len(tv_df_class)

#             tv_json['string'] = tv_df['string'].values.tolist()
#             tv_json['label'] = tv_df['label'].values.tolist()

#             idx += 1
        
#     return dataset_info, train_data, val_data



def get_kobert_tokenizer(pre_trained):
    '''
    hugging API 방식으로 kobert tokenizer 를 불러오는 함수
    '''
    from kobert_tokenizer import KoBERTTokenizer
    tokenizer = KoBERTTokenizer.from_pretrained(pre_trained)
    return tokenizer


def get_vocab(tokenizer):
    '''
    kobert 의 vocab 을 불러오는 함수.
    '''
    import gluonnlp as nlp
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    return vocab


from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
class TextDataset(Dataset):
    '''
    dataset class 를 만드는 class
    '''
    def __init__(self, string_list, label_list, bert_tokenizer, vocab, max_len,
                 pad, pair):
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

        pad: bool
            padding 적용 여부
        
        pair: bool
        '''
        self.transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.string_list = string_list
        self.label_list = label_list

        string_ids_list, valid_length_list, segment_ids_list = self.get_tokenized_value()
        self.string_ids_list = string_ids_list
        self.attention_mask_list = self.gen_attention_mask(string_ids_list, valid_length_list)
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
        string_ids_list = []
        valid_length_list = []
        segment_ids_list = []
        for string in self.string_list:
            try:
                string_ids, valid_length, segment_ids = self.transform([string])
            except TypeError:
                string_ids, valid_length, segment_ids = self.transform(['에러문장'])
                

            string_ids_list.append(string_ids)
            valid_length_list.append(valid_length)
            segment_ids_list.append(segment_ids)
        return string_ids_list, valid_length_list, segment_ids_list
        

    def gen_attention_mask(self, string_ids_list, valid_length_list):
        '''
        문장의 유효한 id 를 구분하기 위한 mask 리스트를 생성하는 mothod

        parameters
        ----------
        string_ids_list: int list, shape=(문장 갯수, 문장 ids 길이)
            token회 된 문장의 id 값 리스트

        valid_length_list: int list, shape=(문장 갯수, 1)
            문장 id 값 중에 유효한 값(padding 이 아닌 값)

        returns
        -------
        attention_mask: int list, shape=(문장 갯수, 문장 ids 길이)
            문장 id 값 중에 유요한 값은 1, 유효하지 않은 값은 0으로 표시한 mask 리스트
        '''
        import torch
        string_ids_list = torch.tensor(string_ids_list)
        attention_mask = torch.zeros_like(string_ids_list)
        for i, v in enumerate(valid_length_list):
            attention_mask[i][:v] = 1
        return attention_mask.float()


    def __getitem__(self, i):
        return self.string_ids_list[i], self.attention_mask_list[i], self.segment_ids_list[i], self.label_list[i]


    def __len__(self):
        return (len(self.label_list))


def get_dataloader(string_list, label_list, batch_size, tokenizer, vocab, max_len, pad, pair, 
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

    vocab: vocab
        토크나이저에서 적용할 vocab

    pad: bool
        padding 적용 여부

    pair: bool
        

    num_workers: int
        데이터 로딩 시 사용할 subprocessor 갯수

    returns
    -------
    dataloader: torch dataloader
        dataloader
    '''
    from torch.utils.data import DataLoader
    # string_list = ['문장1 입니다.', '예시문장 일까요?', '이런 문장도 있지요', '하지만 어쩌면', '이렇게 만들어 볼 수도 있지요']
    # label_list = [0, 2, 0, 1, 2]
    
    dataset = TextDataset(
        string_list=string_list,
        label_list=label_list,
        bert_tokenizer=tokenizer,
        vocab=vocab,
        max_len=max_len,
        pad=True,
        pair=False
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

