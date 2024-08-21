import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
np.random.seed(42)
import pandas as pd
import json
import copy
import random

from share_module import timeutils as tim
from share_module import sensorutils as ssum

sys.path.append('/home/kimyh/python/ai')
from sharemodule import plotutils as plm
from sharemodule import utils as utm

def remove_adnormal(df, sensor_name_list, max_dict, min_dict):
    for sensor_name in sensor_name_list:
        sensor_max = max_dict[sensor_name]
        sensor_min = min_dict[sensor_name]
        df[sensor_name][df[sensor_name] > sensor_max] = np.nan
        df[sensor_name][df[sensor_name] < sensor_min] = np.nan
    return df

# def preprocessing(yesterday, whole_df, onoff_df, system_list, target_name, fuel_type, whole_y_list, norm, info):
#     for system in system_list:
#         system_name = system['name']
#         y_list = system['y_list']
#         y_norm = system['y_norm']
#         if system_name == target_name:
#             break

#     whole_df = ssum.xmy_preprocess(
#         mode=f'{fuel_type}_{system_name}',
#         whole_df=whole_df, 
#         col_list=y_list,
#         yesterday=yesterday, 
#         onoff_df=onoff_df
#     )

#     # 변수 업데이트
#     whole_y_list.extend(y_list)
#     norm.update(y_norm)
#     info[f'{target_name} y list'] = y_list
#     return whole_df, whole_y_list, norm, info


def job(config, info, yesterday):
    
    status, error_info, error_msg = 1, '', ''
    
    # =========================================================================
    # 보조파일 불러오기
    try:
        supports_path = f'{config.root_path}/supports'
        
        # 유효 bus name 정보
        with open(f'{supports_path}/{config.except_json_name}.json', 'r', encoding='utf-8-sig') as f:
            except_bus_name_dict = json.load(f)
        
        # 운행 하지 않는 bus name 정보
        with open(f'{supports_path}/n_exist_bus_name_dict.json', 'r', encoding='utf-8-sig') as f:
            n_exist_bus_name_dict = json.load(f)
        
        # 시스템 정보
        with open(f'{supports_path}/{config.column_json_name}.json', 'r', encoding='utf-8-sig') as f:
            column_json = json.load(f)
        
        for column_data in column_json:
            group = column_data['group']
            if group == config.fuel_type:
                system_list = column_data['system_list']
                break
    
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '보조파일을 불러오는 중 에러가 발생하였습니다.'
        return status, error_info, error_msg
    
    # =========================================================================
    # 데이터 불러오기
    try:
        bus_name_list = except_bus_name_dict[config.fuel_type]
        n_bus_name_list = n_exist_bus_name_dict[config.fuel_type]
        new_bus_name_list = list(set(bus_name_list) - set(n_bus_name_list))
        new_bus_name_list.sort()
    
        data_raw_path = f'{config.root_data_path}/data_raw/{config.fuel_type}'
    
        # 연료 타입 명칭 변경
        if 'diesel' in config.fuel_type:
            new_fuel_type = 'diesel'
        else:
            new_fuel_type = config.fuel_type
    except Exception:
        status = 2
        error_info = utm.get_error_info()
        error_msg = '버스 이름을 불러오는 중 에러가 발생하였습니다.'
        return status, error_info, error_msg
    
    # =========================================================================
    # 시스템 별 진행
    for system in system_list:
        norm_dict = {}
        each_count_dict = {yesterday:{'total':0, 'train':0, 'val':0}}
        #==
        # 테스트용
        if system['name'] != 'DPF':
            continue
        #==
        # =========================================================================
        # 버스별 진행
        for bus_name in tqdm(new_bus_name_list):
            #==
            # 테스트용
            # if bus_name != '경남70아1007':
            #     continue
            #==
            # =========================================================================
            # 데이터 불러오기 & 전처리
            try:
                # 해당 날짜에 운행하지 않은 경우 제외
                try:
                    whole_df = pd.read_csv(f'{data_raw_path}/{bus_name}/{yesterday}.csv', encoding='utf-8-sig')
                except FileNotFoundError:
                    continue
                
                # 누락된 시간 채우기
                whole_df = tim.time_filling(
                    df=whole_df,
                    start=f'{yesterday} 00:00:00',
                    periods=86400
                )
            
                # 결측치 제거
                whole_df = whole_df.dropna(subset=[system['drop']])
                
                # 이상치 제거
                whole_df = remove_adnormal(
                    df=whole_df, 
                    sensor_name_list=system['x_list'], 
                    max_dict=system['x_max'], 
                    min_dict=system['x_min']
                )
                whole_df = remove_adnormal(
                    df=whole_df, 
                    sensor_name_list=system['y_list'], 
                    max_dict=system['y_max'], 
                    min_dict=system['y_min']
                )
                # 너무 짧은 경우 제외
                if len(whole_df) < config.data_size*2:
                    continue
                
                # 전후값 채우기
                whole_df = whole_df.fillna(method='ffill')
                whole_df = whole_df.fillna(method='bfill')
            except Exception:
                status = 2
                error_info = utm.get_error_info()
                error_msg = '데이터 불러오기 & 전처리 중 에러가 발생하였습니다.'
                return status, error_info, error_msg
            # #==
            # for sensor_name in sensor_name_list:
            #     data_ary = whole_df[sensor_name].values
            #     # norm_num = norm[sensor_name]
            #     # sensor_max = max_dict[sensor_name]
            #     # sensor_min = min_dict[sensor_name]
                
            #     plm.draw_plot(
            #         title=f'{sensor_name}',
            #         x=range(len(data_ary)),
            #         y=data_ary,
            #         # y_range=(sensor_min, norm_num),
            #         # y_range=(sensor_min, sensor_max),
            #         # line_color='black',
            #         fig_size=(30, 5),
            #         save_path=f'./image/전처리 후 이미지'
            #     )
            # #==
            
            # =========================================================================
            # 분할
            try:
                i = 0
                seg_data_list = []
                while True:
                    start_idx = i * config.data_interval
                    end_idx = start_idx + config.data_size
                    
                    if end_idx > len(whole_df):
                        break
                    
                    temp_df = whole_df.iloc[start_idx:end_idx, :]
                    
                    x_df = temp_df[system['x_list']]
                    y_df = temp_df[system['y_list']]
                    mark_df = temp_df[system['mark_list']]
                    
                    seg_data_list.append([x_df, y_df, mark_df])
                    i += 1
            except Exception:
                status = 2
                error_info = utm.get_error_info()
                error_msg = '데이터 분할 중 에러가 발생하였습니다.'
                return status, error_info, error_msg
            
            # =========================================================================
            # 저장
            try:
                # train & val 갯수 계산
                total_num = len(seg_data_list)
                train_num = int(total_num * config.train_p/100)
                each_count_dict[yesterday]['total'] += total_num
                each_count_dict[yesterday]['train'] += train_num
                each_count_dict[yesterday]['val'] += total_num - train_num
                
                # 저장
                dataset_save_path = f'{config.root_data_path}/datasets/{config.new_dataset_name}/{new_fuel_type}/{system["name"]}'
                train_idx_ary = np.random.choice(total_num, train_num, replace=False)
                for idx, df_list in enumerate(seg_data_list):
                    x_df = df_list[0]
                    y_df = df_list[1]
                    mark_df = df_list[2]
                    
                    if idx in train_idx_ary:
                        tv = 'train'
                    else:
                        tv = 'val'
                    
                    order = str(idx).zfill(5)
                    os.makedirs(f'{dataset_save_path}/{tv}/x', exist_ok=True)
                    os.makedirs(f'{dataset_save_path}/{tv}/y', exist_ok=True)
                    os.makedirs(f'{dataset_save_path}/{tv}/mark', exist_ok=True)
                    x_df.to_csv(f'{dataset_save_path}/{tv}/x/{bus_name}_{yesterday}_{order}_x.csv')
                    y_df.to_csv(f'{dataset_save_path}/{tv}/y/{bus_name}_{yesterday}_{order}_y.csv')
                    mark_df.to_csv(f'{dataset_save_path}/{tv}/mark/{bus_name}_{yesterday}_{order}_mark.csv')
            except Exception:
                status = 2
                error_info = utm.get_error_info()
                error_msg = '데이터 저장 중 에러가 발생하였습니다.'
                return status, error_info, error_msg
        
        # =================================================================
        # 정보 저장
        try:
            # norm 저장
            norm_dict.update(system["x_norm"])
            norm_dict.update(system["y_norm"])
            norm_dict.update(system["mark_norm"])
            if os.path.isfile(f'{dataset_save_path}/norm.json'):
                with open(f'{dataset_save_path}/norm.json', 'r', encoding='utf-8-sig') as f:
                    pre_norm = json.load(f)
                norm_dict.update(pre_norm)
            os.makedirs(dataset_save_path, exist_ok=True)
            with open(f'{dataset_save_path}/norm.json', 'w', encoding='utf-8-sig') as f:
                json.dump(norm_dict, f, indent='\t', ensure_ascii=False)
            
            # info 저장
            if os.path.isfile(f'{dataset_save_path}/info.json'):
                with open(f'{dataset_save_path}/info.json', 'r', encoding='utf-8-sig') as f:
                    info = json.load(f)
                # info['count']['total']['total'] += each_count_dict[yesterday]['total']
                # info['count']['total']['train'] += each_count_dict[yesterday]['train']
                # info['count']['total']['val'] += each_count_dict[yesterday]['val']
                info['count']['each'].update(each_count_dict)
                temp_name_list = copy.deepcopy(info['bus_name_list'])
                temp_name_list.extend(new_bus_name_list)
                temp_name_list = list(set(temp_name_list))
                temp_name_list.sort()
                info['bus_name_list'] = temp_name_list
            else:
                info['count']['each'] = each_count_dict
                info['bus_name_list'] = new_bus_name_list
                
                # info['count'] = {
                #     'total':{
                #         'total':each_count_dict[yesterday]['total'],
                #         'train':each_count_dict[yesterday]['train'],
                #         'val':each_count_dict[yesterday]['val']
                #     },
                #     'each':each_count_dict
                # }
                
            # 전체 갯수 계산
            # 초기화 한 후 다시 계산하지 않으면 info 를 불러올때마다 갯수가 반복 축적됨
            total_total_num = 0
            total_train_num = 0
            total_val_num = 0
            for _, day_count in info['count']['each'].items():
                total_total_num += day_count['total']
                total_train_num += day_count['train']
                total_val_num += day_count['val']
            info['count']['total']['total'] = total_total_num
            info['count']['total']['train'] = total_train_num
            info['count']['total']['val'] = total_val_num
                
            # 정렬
            info['count']['each'] = dict(sorted(info['count']['each'].items(), 
                    key=lambda item: item[0],
                    reverse=False))
            with open(f'{dataset_save_path}/info.json', 'w', encoding='utf-8-sig') as f:
                json.dump(info, f, indent='\t', ensure_ascii=False)
        except Exception:
            status = 2
            error_info = utm.get_error_info()
            error_msg = '정보 저장 중 에러가 발생하였습니다.'
            print(error_info)
            return status, error_info, error_msg
    
    status = 3
    return status, error_info, error_msg