import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from share_module import timeutils as tim


sys.path.append('/home/kimyh/python/ai')
from sharemodule import plotutils as plm
from sharemodule import utils as utm


import develop_config as config

def main():
    fuel_type = 'diesel_new'
    # explain = 'raw'
    # explain = '결측치제거'
    # explain = '결측치, 이상치제거'
    # explain = '결측치, 이상치제거, filling, minnorm, except 17, only soot'
    # explain = '이상한 데이터'
    explain = '운행 잘 안하는 버스들, only soot'
    do_remove_NaN = True
    do_remove_adnormal = True
    do_fill = True
    only_soot = True
    # except_json_name = 'except_bus_name_17.json'
    except_json_name = 'n_exist_bus_name_dict.json'
    column_json_name = 'column_info_all.json'
    
    # except_type = 'raw'
    except_type = 'bus name'
    # except_type = '이상함'
    # except_type = 'DPF 수치 낮음'
    # except_type = 'DPF 수치 이상함'
    # except_type = 'DPF 일부 굴곡 존재'
    # except_type = '운행기간 너무 짧음'
    
    # ========================================================================
    # 날짜 리스트 불러오기    
    date_list = utm.get_date_list(
        schedule=False,
        year=config.year,
        mon_list=config.mon_list,
        start_day_list=config.start_day_list,
        end_day_list=config.end_day_list
    )

    # ========================================================================
    # column 정보 불러오기
    with open(f'./supports/{column_json_name}', 'r', encoding='utf-8-sig') as f:
        column_info = json.load(f)
    
    for column_data in column_info:
        group = column_data['group']
        if group == fuel_type:
            break
        
    # ========================================================================
    # norm
    sensor_name_list = column_data['sensor_name_list']
    drop = column_data['drop']
    norm = column_data['norm']
    max_dict = column_data['max']
    min_dict = column_data['min']
    
    # ========================================================================
    # 시작 
    
    for date_ in date_list:
        print(date_)
        # ========================================================================
        # 버스 이름 리스트 불러오기
        bus_name_path = f'{config.root_data_path}/data_raw/{fuel_type}'
        if except_type == 'raw':
            bus_name_list = os.listdir(bus_name_path)
        elif except_type == 'bus name':
            with open(f'./supports/{except_json_name}', 'r', encoding='utf-8-sig') as f:
                except_json = json.load(f)
            bus_name_list = except_json[fuel_type]
        else:    
            with open(f'./supports/excepted.json', 'r', encoding='utf-8-sig') as f:
                except_json = json.load(f)
            bus_name_list = except_json[except_type]
        bus_name_list.sort()
        
        n_exist_bus_name_list = []
        print(len(bus_name_list))
        # ========================================================================
        for bus_name in tqdm(bus_name_list):
            
            #==
            # 테스트용
            # if bus_name != '경남70아1007':
            #     continue
            #== 
            
            # ========================================================================
            # 데이터 불러오기
            try:
                whole_df = pd.read_csv(f'{bus_name_path}/{bus_name}/{date_}.csv', encoding='utf-8-sig')
            except FileNotFoundError:
                n_exist_bus_name_list.append(bus_name)
                continue
            
            # ========================================================================
            # 시간 확장
            whole_df = tim.time_filling(
                df=whole_df,
                start=f'{date_} 00:00:00',
                periods=86400
            )
    
            # ========================================================================
            # 속도 기준 NaN 제거
            if do_remove_NaN:
                whole_df = whole_df.dropna(subset=[drop])
    
            # ========================================================================
            # 이상치 --> 결측치
            if do_remove_adnormal:
                for sensor_name in sensor_name_list:
                    sensor_max = max_dict[sensor_name]
                    sensor_min = min_dict[sensor_name]
                    whole_df[sensor_name][whole_df[sensor_name] > sensor_max] = np.nan
                    whole_df[sensor_name][whole_df[sensor_name] < sensor_min] = np.nan
    
            # ========================================================================
            # 전후값 채우기
            if do_fill:
                whole_df = whole_df.fillna(method='ffill')
                whole_df = whole_df.fillna(method='bfill')
                
            # ========================================================================
            # image
            for sensor_name in sensor_name_list:
                data_ary = whole_df[sensor_name].values
                norm_num = norm[sensor_name]
                sensor_max = max_dict[sensor_name]
                sensor_min = min_dict[sensor_name]
                
                if only_soot:
                    if sensor_name != 'ECU_DPFSoot':
                        continue    
                    plm.draw_plot(
                        title=bus_name,
                        x=range(len(data_ary)),
                        y=data_ary,
                        y_range=(sensor_min, norm_num),
                        # y_range=(sensor_min, sensor_max),
                        # line_color='black',
                        fig_size=(30, 5),
                        save_path=f'./image/{explain}/{fuel_type}/{date_}'
                    )
                else:
                    plm.draw_plot(
                        title=f'{sensor_name}',
                        x=range(len(data_ary)),
                        y=data_ary,
                        y_range=(sensor_min, norm_num),
                        # y_range=(sensor_min, sensor_max),
                        # line_color='black',
                        fig_size=(30, 5),
                        save_path=f'./image/{explain}/{fuel_type}/{date_}/{bus_name}'
                    )
                    
        n_exist_bus_name_dict = {
            fuel_type:n_exist_bus_name_list
        }
        with open(f'./supports/n_exist_bus_name_dict.json', 'w', encoding='utf-8-sig') as f:
            json.dump(n_exist_bus_name_dict, f, indent='\t', ensure_ascii=False)
        #==
        # bus_name_list = []
        #==
        

if __name__ == '__main__':
    main()