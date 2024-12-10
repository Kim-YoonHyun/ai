import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils as um	


def read_df(path):
    extention = path.split('.')[-1]
    if extention in ['csv', 'CSV']:
        switch = 'csv'
    elif extention in ['xlsx', 'xls']:
        switch = 'excel'
    elif extention in ['txt']:
        switch = 'txt'
    else:
        raise ValueError(f'{extention}은(는) 잘못되거나 지정되지 않은 확장자입니다.')
    
    if switch == 'csv':
        encoding = 'utf-8-sig'
        while True:
            try:
                data_df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                encoding = 'cp949'
            except pd.errors.ParserError:
                f = open(path, encoding=encoding)
                reader = csv.reader(f)
                csv_list = []
                for line in reader:
                    if len(line) != 38:
                        pass
                    csv_list.append(line)
                f.close()
                data_df = pd.DataFrame(csv_list)
                data_df.columns = data_df.iloc[0].to_list()
                data_df = data_df.drop(index=data_df.index[0])	# 0번째 행을 지움
                break
    if switch == 'excel':
        data_df = pd.read_excel(path)
    if switch == 'txt':
        line_list = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_list.append(line)
        data_df = pd.DataFrame(line_list, columns=['string'])
    return data_df


def utc2kor(df, time_column='time'):
    if df.empty:
        return df
    df[time_column] = df[time_column].astype('str')
    df[time_column] = df[time_column].apply(lambda x: x.replace('T', ' '))
    df[time_column] = df[time_column].apply(lambda x: x.replace('Z', ''))
    
    # UTC 시간을 한국 시간으로 (+9 시간)
    df[time_column] = df[time_column].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df[time_column] = df[time_column].apply(lambda x: x + timedelta(hours=9))
    df[time_column] = df[time_column].astype('str')

    df = df.sort_values(by=time_column,ascending=True)
    
    return df


# def dataframe_preprocessor(df, 
#                            max_dict=None, min_dict=None, 
#                            nan_drop_column=None, 
#                            do_nan_fill_whole=False
#     ):
#     # ========================================================================
#     # # 최대값 초과 --> 이상치 --> 결측치
#     # if max_dict is not None:
#     #     df = maxadnormal2nan(df=df, max_dict=max_dict)
        
#     # # 최소값 미만 --> 이상치 --> 결측치
#     # if min_dict is not None:
#     #     df = minadnormal2nan(df=df, min_dict=min_dict)

#     # ========================================================================
#     # 특정 컬럼 기준 NaN 제거
#     # if nan_drop_column is not None:
#     #     df = drop_nan(
#     #         df=df, 
#     #         base_column=nan_drop_column
#     #     )
        
#     # ========================================================================
#     # 전후값 채우기
#     if do_nan_fill_whole:
#         df = df.fillna(method='ffill')
#         df = df.fillna(method='bfill')
    
#     return df


def adnormal2nan(df, stan_col, max_value=None, min_value=None):
    if max_value is not None:
        df[stan_col][df[stan_col] > max_value] = np.nan
    if min_value is not None:
        df[stan_col][df[stan_col] < min_value] = np.nan
    return df


def time_filling(df, start, end, time_column='time'):
    if df.empty:
        return df

    time_range = pd.date_range(start=start, end=end, freq='S')
    time_range_df = pd.DataFrame(time_range, columns=[time_column])
    time_range_df = time_range_df.astype('str')

    # 합치기
    df = pd.merge(df, time_range_df, how='right')
    return df


def drop_nan(df, stan_col):
    try:
        df = df.dropna(subset=[stan_col])
    except KeyError:
        pass
    return df


def isdfvalid(df, valid_column_list):
    # 유효 컬럼 존재 여부 확인
    try:
        _ = df[valid_column_list]
        return True
    except KeyError:
        return False
    
    
def local_nan_corrention(df, stan_col, stan_nan_num=5):
    '''
    stan_nan_num 에 지정한 수치만큼 반복되는 결측치 구간을
    앞뒤값 채우기로 보정하는 함수
    '''
    stan_ary = df[stan_col].values
    nan_start_idx_list, nan_end_idx_list = um.identify_stan_repeat_section(
        ary=stan_ary,
        stan_value='nan',
        stan_repeat=stan_nan_num,
        mode='below',
        reverse=False
    )
    for nan_si, nan_ei in zip(nan_start_idx_list, nan_end_idx_list):
        df.loc[nan_si-1:nan_ei, stan_col] = df.loc[nan_si-1:nan_ei, stan_col].fillna(method='ffill')
        df.loc[nan_si:nan_ei+1, stan_col] = df.loc[nan_si:nan_ei+1, stan_col].fillna(method='bfill')
    
    return df


def pin2nan(df, stan_col, stan_pin_num=3, stan_nan_num=3):
    '''
    이상치 범위에 속하지 않지만 
    데이터 흐름상 이상치로 볼 필요가 있는 국소 범위의 값들을 결측치로 변경하는 함수
    
    예시: 20, 20, 20, 20, [  1], 20, 20, 20, 1, 1, 2, 1
    결과: 20, 20, 20, 20, [NaN], 20, 20, 20, 1, 1, 2, 1
    '''
    
    # 기준 컬럼 데이터 추출
    stan_ary = df[stan_col].values
    
    # 현재 값에서 이전값을 뺀 데이터 ary 를 생성
    stan_1_list = stan_ary.tolist()
    stan_1_list.insert(0, stan_ary[0])
    stan_1_ary = np.array(stan_1_list)[:-1]
    diff_ary = np.round(stan_ary - stan_1_ary)
    
    # pin 이상치 시작, 끝 idx 추출
    start_idx_list = []
    end_idx_list = []
    start_idx = None
    pre_value = None
    
    for idx, d in enumerate(diff_ary):
        if abs(d) > stan_pin_num:
            
            if pre_value is None:
                start_idx = idx
                pre_value = d
                continue
            if abs(pre_value) == abs(d):
                if idx - start_idx <= stan_nan_num:
                    start_idx_list.append(start_idx)
                    end_idx_list.append(idx)
            start_idx = idx
            pre_value = d

    temp_ary = stan_ary.copy()
    
    # pin idx 가 존재하는 경우 해당 범위를 nan 으로 대체
    if len(start_idx_list) > 0:
        for s, e in zip(start_idx_list, end_idx_list):
            temp_ary[s:e] = np.nan
    
    # nan 의 위치 구하기
    for_fill_start_idx_list, for_fill_end_idx_list = um.identify_stan_repeat_section(
        ary=temp_ary, 
        stan_value='nan',
        stan_repeat=stan_nan_num, 
        mode='below', 
        reverse=False
    )
    
    # 해당 부분을 NaN 값으로 변환
    for fsi, fei in zip(for_fill_start_idx_list, for_fill_end_idx_list):
        df.loc[fsi:fei, stan_col] = np.nan
    return df