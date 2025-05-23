import sys
import os
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import json
from datetime import date, datetime, timedelta

import job_diesel as jd
import job_cng as jc

sys.path.append(os.getcwd())
from share_module import timeutils as tim
from share_module import sensorutils as ssum

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom



develop = True
if develop:
    import develop_config as config
else:
    import config


def get_date_list(schedule, year, mon_list, start_day_list, end_day_list):
    date_list = []
    if schedule:
        yesterday = date.today() - timedelta(1)
        yesterday = str(yesterday)
        date_list = [yesterday]
    else:
        for mon in mon_list:
            start_day = start_day_list[mon-1]
            end_day = end_day_list[mon-1]
            for dd in range(start_day, end_day+1):
                dd = str(dd).zfill(2)
                mm = str(mon).zfill(2)
                date_list.append(f'{year}-{mm}-{dd}')
    return date_list


def make_dataset(yesterday, config, log):
    # basic
    root_path = config.root_path
    root_save_path = config.root_save_path
    random_seed = config.random_seed
    
    # make dataset parameter
    new_dataset_name = config.new_dataset_name
    column_json_name = config.column_json
    except_json_name = config.except_json
    fuel_type = config.md_fuel_type
    bos_num = config.bos_num
    y2x = config.y2x
    
    # 데이터 분할 parameter
    interval_sec = config.interval_sec
    x_seg_sec = config.x_seg_sec
    y_seg_sec = config.y_seg_sec
    train_p = config.train_p
    zero_speed_p = config.zero_speed_p
    
    # info
    info = {
        'dataset_name':new_dataset_name,
        'fuel type':fuel_type,
        'bos num':bos_num,
        'y to x':y2x,
        'interval sec':interval_sec,
        'x length':x_seg_sec,
        'y length':y_seg_sec,
        'zero speed p':zero_speed_p,
        'column json name':column_json_name
    }
    
    random.seed(random_seed)
    print(yesterday)

    # =========================================================================
    # supports 불러오기
    supports_path = f'{root_path}/supports'

    # 버스 id 정보 파일
    with open(f'{supports_path}/{except_json_name}.json', 'r', encoding='utf-8-sig') as f:
        except_bus_id_json = json.load(f)
    except_bus_id_list = except_bus_id_json['except']
    except_bus_id_list.sort()
    
    # 컬럼 정보 파일
    with open(f'{supports_path}/{column_json_name}.json', 'r', encoding='utf-8-sig') as f:
        column_json = json.load(f)

    # 
    for column_data in column_json:
        group = column_data['group']
        if group == fuel_type:
            x_list = column_data['x_list']
            x_norm = column_data['x_norm']
            mark_list = column_data['mark_list']
            mark_norm = column_data['mark_norm']
            system_list = column_data['system_list']
            break
        
    # norm 업데이트
    norm = {}
    norm.update(x_norm)
    norm.update(mark_norm)

    # info
    info['x list'] = x_list
    info['mark list'] = mark_list
    
    # =========================================================================
    # raw 데이터 불러오기
    data_raw_path = f'{root_path}/data_raw/{fuel_type}'
    bus_id_list = os.listdir(data_raw_path)
    bus_id_list.sort()
    whole_df_dict = {}
    print('데이터 불러오기')
    for bus_id in tqdm(bus_id_list):
        
        # 제외 리스트 
        if bus_id in except_bus_id_list:
            continue
        # # ==
        # # 테스트 할때
        # if bus_id not in ['GV01RD02P001T01M13', 'GV01RD02P001T01M11', 'GV01RD02P001T01M17']:
        #    continue
        # # ==
        
        if fuel_type == 'diesel':
            if bus_id in ["GV01RD02P001T05M02", "GV01RD02P001T06M03", "GV01RD02P001T06M04", "GV01RD02P001T08M01", "GV01RD02P001T08M02", "GV01RD02P001T09M01", "GV01RD02P001T09M02", "GV01RD02P001T10M01", "GV01RD02P001T10M02"]:
                continue
        elif fuel_type == 'CNG':
            if bus_id not in ["GV01RD02P001T05M02", "GV01RD02P001T06M03", "GV01RD02P001T06M04", "GV01RD02P001T08M01", "GV01RD02P001T08M02", "GV01RD02P001T09M01", "GV01RD02P001T09M02", "GV01RD02P001T10M01", "GV01RD02P001T10M02"]:
                continue
        else:
            raise ValueError('잘못된 연료타입입니다.')
        
        # 불러오기
        try:
            whole_df = pd.read_csv(f'{data_raw_path}/{bus_id}/{yesterday}.csv', encoding='utf-8-sig')
        except FileNotFoundError:
            continue
        
        # 너무 데이터가 적은 경우 제외 (40분 이하)
        if len(whole_df) <= 2400:
            continue
        
        # 누락된 시간 채우기
        whole_df = tim.time_filling(
            df=whole_df,
            start=f'{yesterday} 00:00:00',
            periods=86400
        )
        whole_df_dict[bus_id] = whole_df

    if len(whole_df_dict) == 0:
        raise ValueError('해당 날짜에 유효한 데이터가 존재하지 않습니다.')
    del bus_id

    # =========================================================================
    dataset_save_path = f'{root_save_path}/datasets/{new_dataset_name}/{fuel_type}'
    if fuel_type == 'diesel':
        jd.job(
            info=info,
            yesterday=yesterday,
            fuel_type=fuel_type,
            whole_df_dict=whole_df_dict,
            x_list=x_list,
            mark_list=mark_list,
            system_list=system_list,
            norm=norm,
            bos_num=bos_num,
            y2x=y2x,
            interval_sec=interval_sec,
            x_seg_sec=x_seg_sec,
            zero_speed_p=zero_speed_p,
            train_p=train_p,
            save_path=dataset_save_path
        )
    elif fuel_type == 'CNG':
        jc.job(
            info=info,
            yesterday=yesterday,
            fuel_type=fuel_type,
            whole_df_dict=whole_df_dict,
            x_list=x_list,
            mark_list=mark_list,
            system_list=system_list,
            norm=norm,
            bos_num=bos_num,
            y2x=y2x,
            interval_sec=interval_sec,
            x_seg_sec=x_seg_sec,
            zero_speed_p=zero_speed_p,
            train_p=train_p,
            save_path=dataset_save_path
        )
        

def main():
    # 로그 파일 생성
    log = lom.get_logger(
        get='RUN_MAKE_DATASET',
        root_path=config.root_path,
        log_file_name='run_make_dataset.log',
        time_handler=True,
        console_display=False,
        schedule=False
    )
    
     # 날짜 기간 생성
    date_list = get_date_list(
        schedule=False,
        year=config.year,
        mon_list=config.mon_list,
        start_day_list=config.start_day_list,
        end_day_list=config.end_day_list
    )
        
    # 실행
    for yesterday in date_list:
        try:
            make_dataset(yesterday, config, log)
        except Exception:
            print('유효한 데이터 없음')
            continue
        
        
if __name__ == '__main__':
    main()