import sys
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json


sys.path.append('/mnt/analysis')
from share_module import dbutils as dbm
from share_module import timeutils as tim


# from mylocalmodules import dbutils as dbm
from mylocalmodules import utils as utm
from mylocalmodules import gpsutils as gpm
# from mylocalmodules import timeutils as tim
# from mylocalmodules import plotutils as plm

def is_data_valid(result, column_list):
    if len(result) == 0:
        return False
    
    whole_df = dbm.query2df(
        result=result, 
        query_type='influxdb2',
        mode='test'
    )
    # 유효하지 않을 경우 넘김
    try:
        _ = whole_df[column_list]
        return True
    except KeyError:
        return False
    
    

def job(yesterday, config, log):
    # root_path = config.root_path
    # fuel_type = config.gdr_fuel_type
    # column_json_name = config.gdr_column_json
    # start_time = config.start_time
    # end_time = config.end_time
    
    # # maria
    # mari_host = config.mari_host
    # mari_port = config.mari_port
    # mari_database = config.mari_database
    # mari_user = config.mari_user
    # mari_password = config.mari_password

    # # influx
    # infl_host = config.infl_host
    # infl_port = config.infl_port
    # infl_database = config.infl_database
    # token = config.token

    # variable initialize
    fuel_type_dict = {'diesel_new':'경유', 'diesel_old':'경유', 'CNG':'CNG'}    
    status, error_info, error_msg = 1, '', ''

    print(yesterday)
    # =========================================================================
    # 필요 컬럼 추출
    try:
        # 서포트 파일
        support_path = f'{config.root_path}/supports'
        with open(f'{support_path}/{config.column_json}.json', 'r', encoding='utf-8-sig') as f:
            column_json = json.load(f)
        
        # 원하는 연료타입인 경우만 진행
        for data in column_json:
            group = data['group']
            x_list = data['x_list']
            mark_list = data['mark_list']
            system_list = data['system_list']
            if group == config.fuel_type:
                break

        # y
        whole_y_list = []
        for system in system_list:
            whole_y_list += system['y_list']
        whole_y_list = list(set(whole_y_list))
        whole_y_list.sort()
        
        # 컬럼 필터링
        valid_column_list = ['time', 'ADC_ACC'] + x_list + whole_y_list + mark_list
        
    except KeyError:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '필요한 컬럼이 존재하지 않습니다.'
        return status, error_info, error_msg
        
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '필요한 컬럼 추출 중 에러가 발생하였습니다.'
        return status, error_info, error_msg
    
    # =========================================================================
    try:
        # query = f'''
        # SELECT reg_no, vin, ve_type FROM cmmn_bus_info
        # WHERE fuel_name="{fuel_name}"
        # '''
        # bus_info_df = dbm.mariadb_query(
        #     host=mari_host, 
        #     port=mari_port, 
        #     database=mari_database, 
        #     user=mari_user, 
        #     password=mari_password, 
        #     query=query, 
        # )
        fuel_name = fuel_type_dict[config.fuel_type]
        
        print('버스 정보 추출')        
        query = f"""SELECT reg_no, vin, ve_type, fuel_name FROM cmmn_bus_info"""
        bus_info_df = dbm.mariadb_query(
            host=config.mari_host,
            port=config.mari_port,
            database=config.mari_database,
            user=config.mari_user,
            password=config.mari_password,
            query=query
        )
        uni_bus_info_df = bus_info_df[bus_info_df['fuel_name'] == fuel_name]
        
        print('센서정보 추출')
        query = f'''
            SELECT ve_type, sensor_group, sensor_code FROM cmmn_sensor_info
        '''
        sensor_info_df = dbm.mariadb_query(
            host=config.mari_host, 
            port=config.mari_port, 
            database=config.mari_database, 
            user=config.mari_user, 
            password=config.mari_password, 
            query=query
        )
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        print(error_info)
        error_msg = 'db 접속 중 에러가 발생하였습니다.'
        return status, error_info, error_msg
    
    
    # =========================================================================
    # raw 데이터 불러오기
    try:
        whole_df_dict = {}
        print(f'모든 {config.fuel_type} 버스 데이터 추출')
        for _, row in tqdm(uni_bus_info_df.iterrows(), total=len(uni_bus_info_df)):
            bus_name = row['reg_no']
            bus_id = row['vin']
            ve_type = row['ve_type']
            
            uni_sensor_info_df = sensor_info_df[sensor_info_df['ve_type'] == ve_type]
            sensor_code_list = uni_sensor_info_df['sensor_code'].values.tolist()
            sensor_code_list.sort()
            
            # 버스 데이터 추출
            # test_query = f"""
            # SELECT TOP 10 * FROM {bus_id}
            # WHERE time >= '{yesterday} 14:00:00' 
            # AND time <= '{yesterday} 15:00:00' tz('Asia/Seoul')
            # """
            test_query = f"""
            SELECT TOP 3 * FROM {bus_id}
            """
            test_result = dbm.influxdb_query(
                config.infl_host, 
                config.infl_port, 
                2, 
                config.infl_database, 
                test_query, 
                token=config.token
            )
            if is_data_valid(test_result, sensor_code_list):
                query = f"""
                SELECT * FROM {bus_id}
                WHERE time >= '{yesterday} {config.start_time}' 
                AND time <= '{yesterday} {config.end_time}' tz('Asia/Seoul')
                """
                result = dbm.influxdb_query(
                    config.infl_host, 
                    config.infl_port, 
                    2, 
                    config.infl_database, 
                    query, 
                    token=config.token
                )
                whole_df = dbm.query2df(result, 'influxdb2')
                whole_df_dict[bus_name] = whole_df

        # 유효한 데이터가 없는 경우
        if len(whole_df_dict) == 0:
            raise ValueError()
    except ValueError:
        status = 2
        error_info = utm.get_error_info()
        print(error_info)
        error_msg = '해당 날짜에 유효한 버스 정보가 존재하지 않습니다.'
        return status, error_info, error_msg

    except Exception:
        status = 2
        error_info = utm.get_error_info()
        print(error_info)
        error_msg = '데이터 불러오는 중 에러가 발생하였습니다.'
        return status, error_info, error_msg

    # =========================================================================
    # 데이터 전처리 진행
    try:
        preprocessed_df_dict = {}
        print('데이터 전처리 진행')
        for bus_name, whole_df in tqdm(whole_df_dict.items()):
            # 필요한 컬럼 데이터만 추출
            try:
                preprocessed_df = whole_df[valid_column_list]
                # preprocessed_df = whole_df
            except KeyError:
                continue
            
            # 시간 형식 통일
            preprocessed_df = tim.utc2kor(preprocessed_df, time_column='time')
            
            # GPS 위경도값 전처리
            preprocessed_df = gpm.lat_lon_preprocess(
                df=preprocessed_df, 
                lat_column='GPS_Latitude', 
                lon_column='GPS_longitude'
            )
            preprocessed_df_dict[bus_name] = preprocessed_df

        # 유효한 데이터가 없는 경우
        if len(preprocessed_df_dict) == 0:
            raise ValueError()
    except ValueError:
        status = 2
        error_info = utm.get_error_info()
        print(error_info)
        error_msg = '해당 날짜에 유효한 버스 정보가 존재하지 않습니다.'
        return status, error_info, error_msg
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '데이터 전처리 중 에러가 발생하였습니다.'
        print(error_info)
        return status, error_info, error_msg
    
    # =========================================================================
    try:
        print('데이터 저장')
        for bus_name, result_df in preprocessed_df_dict.items():
            os.makedirs(f'{config.root_path}/data_raw/{config.fuel_type}/{bus_name}', exist_ok=True)
            result_df.to_csv(f'{config.root_path}/data_raw/{config.fuel_type}/{bus_name}/{yesterday}.csv', index=False, encoding='utf-8-sig')
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '데이터 저장 중 에러가 발생하였습니다.'
        return 2, error_info, error_msg
    

    status = 3
    return status, error_info, error_msg

