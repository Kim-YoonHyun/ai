import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
	

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


def time_filling(df, start, periods, time_column='time'):
    if df.empty:
        return df

    # day = str(df[time_column].values[0]).split(' ')[0]
    time_range = pd.date_range(start=start, periods=periods, freq='S')
    time_range_df = pd.DataFrame(time_range, columns=[time_column])
    time_range_df = time_range_df.astype('str')

    # 합치기
    df = pd.merge(df, time_range_df, how='right')
    return df


def time_filling2(df, start, end):
    if df.empty:
        return df
    df = df.sort_values(by='time')

    time_range = pd.date_range(start=start, end=end, freq='S')
    time_range_df = pd.DataFrame(time_range, columns=['time'])
    time_range_df = time_range_df.astype('str')

    df = pd.merge(df, time_range_df, how='right')
    return df


if __name__ == "__main__":
    pass
