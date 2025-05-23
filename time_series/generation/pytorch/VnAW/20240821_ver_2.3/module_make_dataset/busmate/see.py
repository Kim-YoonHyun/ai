import sys
import os
sys.path.append(os.getcwd())
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append('/home/kimyh/python/ai')
from sharemodule import plotutils as plm
from sharemodule import utils as utm
from sharemodule import timeutils as tum
from sharemodule import dataframeutils as dfum


def image_save(mode, df, image_title, column_name_list, max_dict, min_dict, save_path, focus_start_list, focus_end_list):
    if mode == 'main':
        for sensor_name in column_name_list:
            data_ary = df[sensor_name].values
            y_max = max_dict[sensor_name]
            y_min = min_dict[sensor_name]
            
            plm.draw_plot(
                title=f'{sensor_name}',
                x=range(len(data_ary)),
                y=data_ary,
                title_font_size=20,
                x_font_size=20,
                y_font_size=20,
                y_range=(y_min, y_max),
                # line_color='black',
                fig_size=(30, 5),
                save_path=save_path
            )
    
    if mode == 'sub':
        y_data_ary_list = []
        x_data_ary_list = []
        y_range_list = []
        for sensor_name in column_name_list:
            y_data_ary = df[sensor_name].values
            
            # 양끝에 값 추가 (nan인 경우 이미지에서 생략되는거 방지)
            clean_y = y_data_ary[~np.isnan(y_data_ary)]
            try:
                mi = min(clean_y)
                ma = max(clean_y)
            except ValueError:
                mi = 0
                ma = 1
            if str(y_data_ary[0]) == 'nan':
                y_data_ary[0] = mi
            if str(y_data_ary[-1]) == 'nan':
                y_data_ary[-1] = ma
            
            y_data_ary_list.append(y_data_ary)
            
            x_data_ary = range(len(y_data_ary))
            x_data_ary_list.append(x_data_ary)
            
            y_max = max_dict[sensor_name]
            y_min = min_dict[sensor_name]
            if 'GPS' in sensor_name:
                y_min = mi - 0.1
                y_max = ma + 0.1
            y_range = (y_min, y_max)
            y_range_list.append(y_range)
        
        plm.draw_subplot(
            image_title=image_title,
            sub_row_idx=len(column_name_list),
            sub_col_idx=1,
            title_list=column_name_list,
            x_list=x_data_ary_list,
            y_list=y_data_ary_list,
            title_font_size=20,
            x_font_size=20,
            y_font_size=20,
            y_range_list=y_range_list,
            fig_size=(70, 5*len(column_name_list)),
            focus_start_list=focus_start_list,
            focus_end_list=focus_end_list,
            save_path=save_path
        )


def job(root_data_path, 
        fuel_type, bus_name_list, date_list, column_data, save_path,
        do_remove_adnormal=True, do_remove_nan=False, 
        do_nan_fill_whole=False, do_nan_fill_local=True, local_nan_num=10,
        stop_time=600):
    
    for bus_name in tqdm(bus_name_list):
        #==
        # if bus_name != '경남70아5709':
        #     continue
        #==
        for date_ in date_list:
            #==
            # if date_ != '2024-04-27':
            #     continue
            #==
            # 데이터 불러오기
            try:
                whole_df = pd.read_csv(f'{root_data_path}/data_raw/{fuel_type}/{bus_name}/{date_}.csv', encoding='utf-8-sig')
            except FileNotFoundError:
                continue
            try:
                _ = whole_df[list(column_data['max'].keys())]
            except KeyError:
                continue
            
            # 데이터 전처리
            whole_df = dfum.dataframe_preprocessor(
                df=whole_df,
                max_dict=column_data['max'], 
                min_dict=column_data['min'], 
                start_time=f'{date_} 00:00:00', 
                end_time=f'{date_} 23:59:59',
                drop_col=column_data['drop'],
                do_remove_adnormal=do_remove_adnormal,
                do_remove_nan=do_remove_nan,
                do_nan_fill=do_nan_fill_whole
            )
            #==
            temp = whole_df['GPS_Latitude'].values
            clean_temp = temp[~np.isnan(temp)]
            #==
            # 미세한 결측치 구간 산출 & 보정
            if do_nan_fill_local:
                nan_start_idx_list, nan_end_idx_list = utm.identify_nan_section(
                    ary=whole_df['GPS_Latitude'].values,
                    stan_num=local_nan_num,
                    mode='below',
                    reverse=False
                )
                for nan_si, nan_ei in zip(nan_start_idx_list, nan_end_idx_list):
                    whole_df.iloc[nan_si-1:nan_ei+1] = whole_df.iloc[nan_si-1:nan_ei+1].fillna(method='ffill')
                    whole_df.iloc[nan_si-1:nan_ei+1] = whole_df.iloc[nan_si-1:nan_ei+1].fillna(method='bfill')
            
            # GPS 기준 멈춘구간 산정
            start_idx_list, end_idx_list = utm.identify_repeat_section(
                ary=whole_df['GPS_Latitude'].values,
                stan_num=stop_time
            )
            
            # 이미지 저장
            image_save(
                mode='sub',
                df=whole_df, 
                image_title=f'DPF_{date_}',
                column_name_list=column_data['sensor_name_list'], 
                max_dict=column_data['plot_max'], 
                min_dict=column_data['plot_min'], 
                save_path=f'{save_path}/{bus_name}',
                focus_start_list=start_idx_list,
                focus_end_list=end_idx_list
            )  
        
        
def main():
    year = 2024
    mon_list = [4, 5, 6, 7]
    #  month order = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    start_day_list = [22, 19, 24, 18,  1, 24,  1,  1,  1,  1,  1,  1]
    end_day_list   = [22, 19, 30, 30, 31, 30, 16, 31, 30, 31, 30, 31]
    
    fuel_type = 'diesel_new'
    column_json_name = 'column_info_all_04.json'
    root_path = '/home/kimyh/python/project/busmate'
    root_data_path = '/data/busmate'
    
    # explain = '결측치, 이상치제거, filling, minnorm, except 17, only soot'
    # explain = '테스트'
    stop_time = 600
    do_remove_adnormal = True
    do_remove_nan = False
    do_nan_fill_whole = False
    do_nan_fill_local = True
    local_nan_num = 10
    # only_soot = True
    
    valid_bus_name_json = 'valid_bus_name_18.json'
    wrong_bus_name_json = 'wrong_bus_name_02.json'
    except_bus_name_json = 'except_bus_name_01.json'
    
    # ========================================================================
    # 날짜 리스트 불러오기    
    date_list = tum.get_date_list(
        schedule=False,
        year=year,
        mon_list=mon_list,
        start_day_list=start_day_list,
        end_day_list=end_day_list
    )

    # ========================================================================
    # column 정보 불러오기
    with open(f'./supports/{column_json_name}', 'r', encoding='utf-8-sig') as f:
        column_info = json.load(f)
    
    # column
    for column_data in column_info:
        group = column_data['group']
        if group == fuel_type:
            break

    # ========================================================================
    # 버스 이름 정보 불러오기
    whole_bus_name_list = os.listdir(f'{root_data_path}/data_raw/{fuel_type}')
        
    with open(f'./supports/{valid_bus_name_json}', 'r', encoding='utf-8-sig') as f:
        valid_bus_name_dict = json.load(f)
    valid_bus_name_list = valid_bus_name_dict[fuel_type]
    
    with open(f'./supports/{wrong_bus_name_json}', 'r', encoding='utf-8-sig') as f:
        wrong_bus_name_dict = json.load(f)
    
    with open(f'./supports/{except_bus_name_json}', 'r', encoding='utf-8-sig') as f:
        except_bus_name_dict = json.load(f)
    except_bus_name_list = except_bus_name_dict[fuel_type]
        
    # ========================================================================
    # norm
    # sensor_name_list = column_data['sensor_name_list']
    # drop_col = column_data['drop']
    # norm = column_data['norm']
    # max_dict = column_data['max']
    # min_dict = column_data['min']
    
    # ========================================================================
    # 시작 
    print('문제되는 버스 이미지 저장')
    for key, wrong_bus_name_list in wrong_bus_name_dict.items():
        print(key)
        #==
        if key == '미변동구간존재':
            continue
        #==
        save_path = f'{root_path}/image/wrong4/{key}'
        job(
            root_data_path=root_data_path, 
            fuel_type=fuel_type, 
            bus_name_list=wrong_bus_name_list, 
            date_list=date_list, 
            column_data=column_data, 
            do_remove_adnormal=do_remove_adnormal,
            do_remove_nan=do_remove_nan,
            do_nan_fill_whole=do_nan_fill_whole, 
            do_nan_fill_local=do_nan_fill_local,
            local_nan_num=local_nan_num,
            stop_time=stop_time,
            save_path=save_path
        )
        
    # 정상
    print('정상 버스 이미지 저장')
    save_path = f'{root_path}/image/vaild4/'
    job(
        root_data_path=root_data_path, 
        fuel_type=fuel_type, 
        bus_name_list=valid_bus_name_list, 
        date_list=date_list, 
        column_data=column_data, 
        do_remove_adnormal=do_remove_adnormal,
        do_remove_nan=do_remove_nan,
        do_nan_fill_whole=do_nan_fill_whole, 
        do_nan_fill_local=do_nan_fill_local,
        local_nan_num=local_nan_num,
        stop_time=stop_time,
        save_path=save_path
    )
    
    # 제외
    print('제외 버스 이미지 저장')
    save_path=f'{root_path}/image/except4/'
    job(
        root_data_path=root_data_path, 
        fuel_type=fuel_type, 
        bus_name_list=except_bus_name_list, 
        date_list=date_list, 
        column_data=column_data, 
        do_remove_adnormal=do_remove_adnormal,
        do_remove_nan=do_remove_nan,
        do_nan_fill_whole=do_nan_fill_whole, 
        do_nan_fill_local=do_nan_fill_local,
        local_nan_num=local_nan_num,
        stop_time=stop_time,
        save_path=save_path
    )
    
    # print()
    sys.exit()

    
if __name__ == '__main__':
    main()