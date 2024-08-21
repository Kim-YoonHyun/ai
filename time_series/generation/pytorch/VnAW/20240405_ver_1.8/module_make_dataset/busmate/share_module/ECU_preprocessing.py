import pandas as pd
import numpy as np
from itertools import groupby

def ECU_preprocessing(df):
    # group_size 보다 작은 valid행은 1->0, 0->1로 변환 
    group_size = 300
    # DPFSoot량이 전 후 행과 20 이상 차이나면 둘 다 결측치로 설정
    dpf_gap = 20
    # threshold 이상 모든 열이 연속된 결측치를 검사
    threshold = 30
    
    df = df[['time','ADC_ACC','ECU_DOCGasTemperature_After','ECU_DOCGasTemperature_Before','ECU_DPFSoot','ECU_HCIDosing_State']]
    colums = ['ADC_ACC','ECU_DOCGasTemperature_After','ECU_DOCGasTemperature_Before','ECU_DPFSoot','ECU_HCIDosing_State']
    columns_to_interpolate = ['ECU_HCIDosing_State', 'ECU_DOCGasTemperature_After','ECU_DOCGasTemperature_Before','ECU_DPFSoot']
    df['valid'] = np.nan

    ## ======================================================================================================================
    
    # 모든 데이터가 결측치인 행 -> valid 0
    for column in colums:  
        # 처음으로 값이 들어오는 데이터 전까지 결측치 확인
        first_valid_index = df.dropna(how='all').index[0] if not df.dropna(how='all').empty else None
        if first_valid_index is not None and first_valid_index >= threshold:
            df.loc[:first_valid_index-1, ['valid', 'ADC_ACC']] = 0

        # 데이터프레임 끝 부분 처리
        last_valid_index = df.dropna(how='all').index[-1] if not df.dropna(how='all').empty else None
        if last_valid_index is not None and (len(df) - last_valid_index - 1) >= 0:
            df.loc[last_valid_index+1:, ['valid', 'ADC_ACC']] = 0
            
    # ADC_ACC는 이전 행의 유효한 값으로 처리(임시)
    df['ADC_ACC'] = df['ADC_ACC'].fillna(method='ffill')
    # ADC_ACC가 0인 행 -> valid 0
    df.loc[df['ADC_ACC']==0, 'valid'] = 0
    # 30행 이상 모든 열이 연속된 결측치인 경우 -> valid 0
    for column in columns_to_interpolate:
        non_nan_indices = df[column].notnull().astype(int)
        non_nan_indices_group = non_nan_indices.diff().ne(0).cumsum()
        missing_counts = df[column].isnull().groupby(non_nan_indices_group).sum()
        long_missing_groups = missing_counts[missing_counts >= threshold].index
        for group in long_missing_groups:
            df.loc[non_nan_indices_group == group, 'valid'] = 0

   
    # 센서 상한과 하한을 기준으로 1차 전처리 
    df.loc[(df['ECU_DOCGasTemperature_Before'] > 1000)|(df['ECU_DOCGasTemperature_Before'] < 0), 'ECU_DOCGasTemperature_Before'] = np.nan
    df.loc[(df['ECU_DOCGasTemperature_After'] > 1000)|(df['ECU_DOCGasTemperature_After'] < 0), 'ECU_DOCGasTemperature_After'] = np.nan
    df.loc[(df['ECU_DPFSoot'] > 200)|(df['ECU_DPFSoot'] < 0), 'ECU_DPFSoot'] = np.nan
    df.loc[(df['ECU_HCIDosing_State'] != 0)&(df['ECU_HCIDosing_State'] != 1), 'ECU_HCIDosing_State'] = np.nan
    
    
    # 각 열에 대해 첫 번째 값이 결측치인 경우, 유효한 값이 나오기 전까지만 해당 값으로 보간
    filtered_df = df[df['valid'] != 0].copy()
    for col in columns_to_interpolate:
        if pd.isnull(filtered_df[col].iloc[0]):
            first_valid_index = filtered_df[col].first_valid_index()
            if first_valid_index is not None:
                first_valid_value = filtered_df[col].loc[first_valid_index]
                first_valid_pos = filtered_df.index.get_loc(first_valid_index)
                filtered_df[col].iloc[:first_valid_pos] = first_valid_value
    for col in columns_to_interpolate:
        df.loc[df['valid'] != 0, col] = filtered_df[col]
        
    # 나머지 행 -> valid 1
    df.loc[df['valid']!=0, 'valid'] = 1
    
    # ECU_HCIDosing_State는 이전 행의 유효한 값으로 결측치 처리 / ECU_DPFSoot는 이전 행보다 큰 값이 있는 경우를 확인하고, 이상치 처리 후 선형 보간
    # ECU_DPFSoot 가 1행에 20 이상 차이나는 경우 전 후 값 모두 결측치 처리
    df['ECU_HCIDosing_State'] = df['ECU_HCIDosing_State'].fillna(method='ffill')
    v_idx=[]
    for idx in df.index[:-1]:
        if df.at[idx, 'ECU_HCIDosing_State'] == 1 and df.at[idx + 1, 'ECU_HCIDosing_State'] == 0:
            nan_indices_within_5_rows = df.loc[idx-5:idx+5, 'ECU_DPFSoot'][pd.isna(df['ECU_DPFSoot'])].index
            if not nan_indices_within_5_rows.empty:
                for nan_idx in nan_indices_within_5_rows:
                    v_idx.append(nan_idx)
                    prev_valid = df.loc[:nan_idx-1, 'ECU_DPFSoot'].dropna().tail(1)
                    next_valid = df.loc[nan_idx+1:, 'ECU_DPFSoot'].dropna().head(1)
                    if not prev_valid.empty and not next_valid.empty and abs(prev_valid.values[0] - next_valid.values[0]) > dpf_gap:
                        df.at[nan_idx, 'ECU_DPFSoot'] = 0
                        
    for idx in range(1, len(df) - 1):
        if idx not in v_idx and (idx - 1) not in v_idx:
            if pd.notna(df.at[idx, 'ECU_DPFSoot']) and pd.notna(df.at[idx + 1, 'ECU_DPFSoot']):
                if abs(df.at[idx, 'ECU_DPFSoot'] - df.at[idx + 1, 'ECU_DPFSoot']) > dpf_gap:
                    df.at[idx, 'ECU_DPFSoot'] = np.nan
                    df.at[idx + 1, 'ECU_DPFSoot'] = np.nan
                    
    # 선형보간
    df['ECU_DPFSoot'] = df['ECU_DPFSoot'].interpolate(method='linear')
    df['ECU_DOCGasTemperature_After'] = df['ECU_DOCGasTemperature_After'].interpolate(method='linear')
    df['ECU_DOCGasTemperature_Before'] = df['ECU_DOCGasTemperature_Before'].interpolate(method='linear')
    df['ECU_DOCGasTemperature_After'] = df['ECU_DOCGasTemperature_After'].round(1)
    df['ECU_DOCGasTemperature_Before'] = df['ECU_DOCGasTemperature_Before'].round(1)
    
    # 쪼개진 valid 합치기 
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 1) & (df['group_size'] <=group_size), 'valid'] = 0
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 0) & (df['group_size'] <=group_size), 'valid'] = 1
    df.drop(['group', 'group_size'], axis=1, inplace=True)
    
    # HCI가 비활성화인 경우, ECU_DOCGasTemperature_Before 값이 400~500도 사이인 행만 필터링
    def check_conditions(df):
        filtered = df[(df['ECU_HCIDosing_State'] != 1) & (df['ECU_DOCGasTemperature_Before'].between(400, 500))]
        # 조건을 만족하는 행의 수가 1200회 이상인지 확인
        if len(filtered) >= 1200:
            # 해당되면 모든 'valid' 열 값을 0으로 설정
            df['valid'] = 0
    check_conditions(df)
    
    return df