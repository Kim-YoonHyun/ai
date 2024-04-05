import pandas as pd
import numpy as np

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