import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from itertools import groupby

sys.path.append('/mnt/analysis')
from share_module import timeutils as tim

def xmy_preprocess(mode, whole_df, col_list, onoff_df=None, yesterday=None, col_max=None, col_min=None):
    if mode == 'diesel_제네레이터':
        # print('diesel 제네레이터 전처리')
        col_df = Voltage_preprocessing(whole_df)
    elif mode == 'diesel_냉각수온도':
        # print('diesel 냉각수온도 전처리')
        col_df = Coolant_preprocessing(whole_df)
    elif mode == 'diesel_DPF':
        # print('diesel DPF 전처리')
        col_df = ECU_preprocessing(whole_df)
    elif mode == 'diesel_SCR':
        # print('diesel SCR 전처리')
        col_df = SCR_preprocessing(whole_df)
    elif mode == 'CNG_제네레이터':
        print(mode)
    elif mode == 'CNG_냉각수온도':
        print(mode)
    elif mode == 'CNG_실린더실화':
        print(mode)
    elif mode == 'CNG_연료탱크':
        print(mode)
    else:
        # print('그 외')
        # 데이터 전처리 처리
        col_df = whole_df[['time'] + col_list]
        for col in col_list:
            col_df.loc[(col_df[col] > col_max)|(col_df[col] < col_min), col] = np.nan
        col_df = col_df.fillna(method='ffill')
        col_df = col_df.fillna(method='bfill')

    if onoff_df is not None:
        # 유효값 추출
        col_df = col_df[onoff_df['valid'] == 1]
        
        # 누락 시간 채우기
        col_df = tim.time_filling(
            df=col_df,
            start=f'{yesterday} 00:00:00',
            periods=86400
        )
    col_df = col_df[col_list]

    # 기존의 데이터프레임에 전처리된 값 입력
    whole_df[col_list] = col_df
    
    return whole_df


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
        first_valid_index = df[column].first_valid_index()
        if first_valid_index is not None and first_valid_index >= threshold:
            df.loc[:first_valid_index-1, 'valid'] = 0

        # 마지막 데이터로부터 데이터의 끝까지 모두 결측치인 경우 확인
        last_valid_index = df[column].last_valid_index()
        if last_valid_index is not None and (len(df) - last_valid_index - 1) >= 0:
            df.loc[last_valid_index+1:, 'valid'] = 0
            
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


import pandas as pd
import numpy as np

def SCR_preprocessing(df):
    # group_size 보다 작은 valid행은 1->0, 0->1로 변환 
    group_size = 300
    # DPFSoot량이 전 후 행과 20 이상 차이나면 둘 다 결측치로 설정
    dpf_gap = 20
    # threshold 이상 모든 열이 연속된 결측치를 검사
    threshold = 30
    
    colums = ['ADC_ACC','SCR_CatalystNOx_Before','SCR_CatalystNOx_After','SCR_CatalystTemperature_Before','SCR_DosingModuleDuty']
    columns_to_interpolate = ['SCR_CatalystNOx_Before','SCR_CatalystNOx_After','SCR_CatalystTemperature_Before','SCR_DosingModuleDuty']
    df['valid'] = np.nan
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
    df.loc[df['valid']!=0, 'valid'] = 1
            
    # 30행 이상 모든 열이 연속된 결측치인 경우 -> valid 0
    for column in colums:
        non_nan_indices = df[column].notnull().astype(int)
        non_nan_indices_group = non_nan_indices.diff().ne(0).cumsum()
        missing_counts = df[column].isnull().groupby(non_nan_indices_group).sum()
        long_missing_groups = missing_counts[missing_counts >= threshold].index
        for group in long_missing_groups:
            df.loc[non_nan_indices_group == group, ['valid', 'ADC_ACC']] = 0
            
        
    # 센서 상한과 하한을 기준으로 1차 전처리 
    df.loc[(df['SCR_CatalystNOx_Before'] > 10000)|(df['SCR_CatalystNOx_Before'] < -500), 'SCR_CatalystNOx_Before'] = np.nan
    df.loc[(df['SCR_CatalystNOx_After'] > 10000)|(df['SCR_CatalystNOx_After'] < -500), 'SCR_CatalystNOx_After'] = np.nan
    df.loc[(df['SCR_CatalystTemperature_Before'] > 1000)|(df['SCR_CatalystTemperature_Before'] < 0), 'SCR_CatalystTemperature_Before'] = np.nan
    df.loc[(df['SCR_DosingModuleDuty'] > 150)&(df['SCR_DosingModuleDuty'] < 0), 'SCR_DosingModuleDuty'] = np.nan

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

    # SCR_CatalystNOx_Before랑 SCR_CatalystNOx_After가 -100 이하면 np.nan 
    df.loc[df['SCR_CatalystNOx_Before']<-100,'SCR_CatalystNOx_Before'] = np.nan
    df.loc[df['SCR_CatalystNOx_After']<-100,'SCR_CatalystNOx_After'] = np.nan


    # SCR_CatalystTemperature_Before는 선형보간, 소수점 첫째자리에서 반올림
    df['SCR_CatalystTemperature_Before'] = df['SCR_CatalystTemperature_Before'].interpolate(method='linear')
    df['SCR_CatalystTemperature_Before'] = df['SCR_CatalystTemperature_Before'].round(1)
    df['SCR_CatalystNOx_Before'] = df['SCR_CatalystNOx_Before'].interpolate(method='linear')
    df['SCR_CatalystNOx_After'] = df['SCR_CatalystNOx_After'].interpolate(method='linear')
    df['SCR_DosingModuleDuty'] = df['SCR_DosingModuleDuty'].interpolate(method='linear')
    df['SCR_CatalystNOx_Before'] = df['SCR_CatalystNOx_Before'].round(1)
    df['SCR_CatalystNOx_After'] = df['SCR_CatalystNOx_After'].round(1)
    df['SCR_DosingModuleDuty'] = df['SCR_DosingModuleDuty'].round(2)

    # 'SCR_CatalystNOx_After'가 2000 이상의 값을 가진 시점에서 ADC_ACC가 0이 될 때 까지 전부 valid 0
    condition = df['SCR_CatalystNOx_After'] >= 2000
    indices_2000 = df.index[condition]
    indices_adc_acc_1 = df.index[df['ADC_ACC'] == 1]

    for idx in indices_2000:
        if idx in indices_adc_acc_1:
            # 현재 인덱스에서 이전 인덱스 중 'ADC_ACC'가 0인 최초 위치 찾기
            zero_indices = df.index[(df.index < idx) & (df['ADC_ACC'] == 0)]
            if not zero_indices.empty:
                last_zero_idx = zero_indices[-1]
                # 해당 범위 내 'valid'를 0으로 설정
                df.loc[last_zero_idx:idx, 'valid'] = 0

                
    # SCR_CatalystNOx_Before 또는 SCR_CatalystNOx_After가 2000 이상 달성 이후 30행까지 valid 0 (최초 값 수신 이후 15~30초간 농도가 크게 나타나는 현상 발생함)
    def NOx_anomal_range(df):
        condition_mask = (df['ADC_ACC'] == 1) & ((df['SCR_CatalystNOx_After'] > 2000) | (df['SCR_CatalystNOx_Before'] > 2000))
        valid_indices = df.index[condition_mask]
        df['temp_valid'] = df['valid']
        
        for idx in valid_indices:
            end_idx = min(idx + 30, len(df) - 1) 
            df.iloc[idx:end_idx+1, df.columns.get_loc('temp_valid')] = 0
        df['valid'] = df['temp_valid']
        df.drop('temp_valid', axis=1, inplace=True)
        
        return df
    df = NOx_anomal_range(df)


    # np.nan 넣은 부분 보정
    df['SCR_CatalystTemperature_Before'] = df['SCR_CatalystTemperature_Before'].interpolate(method='linear')
    df['SCR_CatalystTemperature_Before'] = df['SCR_CatalystTemperature_Before'].round(1)
    df['SCR_CatalystNOx_Before'] = df['SCR_CatalystNOx_Before'].interpolate(method='linear')
    df['SCR_CatalystNOx_After'] = df['SCR_CatalystNOx_After'].interpolate(method='linear')
    df['SCR_DosingModuleDuty'] = df['SCR_DosingModuleDuty'].interpolate(method='linear')
    df['SCR_CatalystNOx_Before'] = df['SCR_CatalystNOx_Before'].round(1)
    df['SCR_CatalystNOx_After'] = df['SCR_CatalystNOx_After'].round(1)
    df['SCR_DosingModuleDuty'] = df['SCR_DosingModuleDuty'].round(2)


    # 쪼개진 valid 합치기 
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 1) & (df['group_size'] <=group_size), 'valid'] = 0
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 0) & (df['group_size'] <=group_size), 'valid'] = 1
    df.drop(['group', 'group_size'], axis=1, inplace=True)


    # 1200행 미만의 valid 1 행은 valid 0으로 변경  
    valid_indices = df.index[df['valid'] == 1].to_series()
    groups = (valid_indices.diff() > 1).cumsum()
    group_sizes = valid_indices.groupby(groups).size()
    small_groups = group_sizes[group_sizes < 1200].index
    for group in small_groups:
        idx_to_update = valid_indices[groups == group].index
        df.loc[idx_to_update, 'valid'] = 0
        
    
    return df


def Voltage_preprocessing(df):
    # group_size 보다 작은 valid행은 1->0, 0->1로 변환 
    group_size = 300
    # threshold 이상 모든 열이 연속된 결측치를 검사
    threshold = 30
    # 고장 기준 센서값 (고장코드에서 삭제)
    vol_high = 29
    vol_safe = vol_high * 0.95 # 95% 미만 주의/위험
    # 기준 엔진속도 
    speed = 480
    
    colums = ['ADC_ACC','ECU_BatteryVoltage','VCU_EngineSpeed']
    columns_to_interpolate = ['ECU_BatteryVoltage','VCU_EngineSpeed']
    df['valid'] = np.nan
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
    df['ADC_ACC'] = df['ADC_ACC'].fillna(method='bfill')
    
    # ADC_ACC가 0인 행 -> valid 0
    df.loc[df['ADC_ACC']==0, 'valid'] = 0
    
    # VCU_EngineSpeed가 기준속도 이하인 행 -> valid 0 (고장코드에서는 삭제)
    df.loc[df['VCU_EngineSpeed']<speed , 'valid'] = 0     
    
    # 전압 95% 미만으로 떨어지는 경우 -> valid 0 (고장코드에서는 삭제)
    df.loc[df['ECU_BatteryVoltage']<vol_safe,'valid'] = 0
    
    # 30행 이상 모든 열이 연속된 결측치인 경우 -> valid 0
    for column in colums:
        non_nan_indices = df[column].notnull().astype(int)
        non_nan_indices_group = non_nan_indices.diff().ne(0).cumsum()
        missing_counts = df[column].isnull().groupby(non_nan_indices_group).sum()
        long_missing_groups = missing_counts[missing_counts >= threshold].index
        for group in long_missing_groups:
            df.loc[non_nan_indices_group == group, ['valid', 'ADC_ACC']] = 0
            
    # 센서 상한과 하한을 기준으로 1차 전처리 
    df.loc[(df['ECU_BatteryVoltage'] > 35)|(df['ECU_BatteryVoltage'] < 0), 'ECU_BatteryVoltage'] = np.nan
    df.loc[(df['VCU_EngineSpeed'] > 5000)|(df['VCU_EngineSpeed'] < 0), 'VCU_EngineSpeed'] = np.nan

    # ECU_BatteryVoltage 결측치 이전 값으로 보간
    mask = ((df['ADC_ACC'] == 1)&(df['VCU_EngineSpeed']>speed))
    df.loc[mask, 'ECU_BatteryVoltage'] = df.loc[mask, 'ECU_BatteryVoltage'].fillna(method='bfill')
    # EngineSpeed는 선형보간
    df.loc[mask, 'VCU_EngineSpeed'] = df.loc[mask, 'VCU_EngineSpeed'].interpolate(method='linear')
    
    
    # 나머지 행 -> valid 1
    df.loc[df['valid']!=0, 'valid'] = 1
    
    # 쪼개진 valid 합치기 
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 1) & (df['group_size'] <=group_size), 'valid'] = 0
    # df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    # df['group_size'] = df.groupby('group')['valid'].transform('size')
    # df.loc[(df['valid'] == 0) & (df['group_size'] <=group_size), 'valid'] = 1
    df.drop(['group', 'group_size'], axis=1, inplace=True)


    # 1200행 미만의 valid 1 행은 valid 0으로 변경  
    valid_indices = df.index[df['valid'] == 1].to_series()
    groups = (valid_indices.diff() > 1).cumsum()
    group_sizes = valid_indices.groupby(groups).size()
    small_groups = group_sizes[group_sizes < 1200].index
    for group in small_groups:
        idx_to_update = valid_indices[groups == group].index
        df.loc[idx_to_update, 'valid'] = 0
        
    return df


def Coolant_preprocessing(df):
    # group_size 보다 작은 valid행은 1->0, 0->1로 변환 
    group_size = 300
    # threshold 이상 모든 열이 연속된 결측치를 검사
    threshold = 30
    # 기준 엔진속도 
    speed = 480
    colums = ['ADC_ACC','VCU_CoolantTemperature','VCU_EngineSpeed']
    columns_to_interpolate = ['VCU_CoolantTemperature','VCU_EngineSpeed']
    df['valid'] = np.nan
    mask = ((df['ADC_ACC'] == 1))
    
    
    #===================================================================================
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
    df['ADC_ACC'] = df['ADC_ACC'].fillna(method='bfill')
    
    # ADC_ACC가 0인 행 -> valid 0
    df.loc[df['ADC_ACC']==0, 'valid'] = 0
    
    # 30행 이상 모든 열이 연속된 결측치인 경우 -> valid 0
    for column in colums:
        non_nan_indices = df[column].notnull().astype(int)
        non_nan_indices_group = non_nan_indices.diff().ne(0).cumsum()
        missing_counts = df[column].isnull().groupby(non_nan_indices_group).sum()
        long_missing_groups = missing_counts[missing_counts >= threshold].index
        for group in long_missing_groups:
            df.loc[non_nan_indices_group == group, ['valid', 'ADC_ACC']] = 0
            
    # 센서 상한과 하한을 기준으로 1차 전처리 
    df.loc[(df['VCU_CoolantTemperature'] > 200)|(df['VCU_CoolantTemperature'] < -40), 'VCU_CoolantTemperature'] = np.nan
    df.loc[(df['VCU_EngineSpeed'] > 5000)|(df['VCU_EngineSpeed'] < 0), 'VCU_EngineSpeed'] = np.nan

    
    # # 냉각수 온도가 이전 값과 5 이상 차이나는 경우 np.nan으로 변경
    df['VCU_CoolantTemperature'] = np.where(
    abs(df['VCU_CoolantTemperature'].fillna(method='ffill') - df['VCU_CoolantTemperature'].fillna(method='ffill').shift(1)) >= 5,
    np.where(
        df['VCU_CoolantTemperature'].notna(),  # 원래 NaN이 아닌 값을 유지하기 위한 조건
        np.nan,  # 조건을 만족하면 NaN으로 설정
        df['VCU_CoolantTemperature']  # 원래 NaN인 값은 그대로 NaN으로 유지
    ),
    df['VCU_CoolantTemperature']  # 조건을 만족하지 않으면 원래 값을 유지
    )
    
    # VCU_CoolantTemperature 결측치 보간
    df.loc[mask, 'VCU_CoolantTemperature'] = df.loc[mask, 'VCU_CoolantTemperature'].fillna(method='bfill')  
    
    # 나머지 행 -> valid 1
    df.loc[df['valid']!=0, 'valid'] = 1
    
    # 쪼개진 valid 합치기 
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 1) & (df['group_size'] <=group_size), 'valid'] = 0
    df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
    df['group_size'] = df.groupby('group')['valid'].transform('size')
    df.loc[(df['valid'] == 0) & (df['group_size'] <=group_size), 'valid'] = 1
    df.drop(['group', 'group_size'], axis=1, inplace=True)


    # 1200행 미만의 valid 1 행은 valid 0으로 변경  
    valid_indices = df.index[df['valid'] == 1].to_series()
    groups = (valid_indices.diff() > 1).cumsum()
    group_sizes = valid_indices.groupby(groups).size()
    small_groups = group_sizes[group_sizes < 1200].index
    for group in small_groups:
        idx_to_update = valid_indices[groups == group].index
        df.loc[idx_to_update, 'valid'] = 0
        
    return df