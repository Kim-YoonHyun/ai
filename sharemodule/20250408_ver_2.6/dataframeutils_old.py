import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')
	

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


def dataframe_preprocessor(df, max_dict=None, min_dict=None, 
                           start_time=None, end_time=None, time_column='time', drop_col=None, 
                           do_remove_adnormal=False, do_remove_nan=False, do_nan_fill=False):
    # ========================================================================
    # 이상치 --> 결측치
    if max_dict is not None:
    # if do_remove_adnormal:
        df = adnormal2nan(
            df=df,
            max_dict=max_dict,
            min_dict=min_dict
        )
    # ========================================================================
    # 시간 확장
    df = time_filling(
        df=df,
        start=start_time,
        end=end_time,
        time_column=time_column
    )
    # ========================================================================
    # 특정 컬럼 기준 NaN 제거
    if do_remove_nan:
        df = drop_nan(
            df=df, 
            base_column=drop_col
        )
        
    # ========================================================================
    # 전후값 채우기
    if do_nan_fill:
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
    
    return df


def maxadnormal2nan(df, max_dict):
    for name, max_value in max_dict.items():
        df[name][df[name] > max_value] = np.nan
    for name, min_value in min_dict.items():
        df[name][df[name] < min_value] = np.nan
    return df


def minadnormal2nan(df, min_dict):
    for name, min_value in min_dict.items():
        df[name][df[name] < min_value] = np.nan
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


def drop_nan(df, base_column):
    try:
        df = df.dropna(subset=[base_column])
    except KeyError:
        pass
    return df

