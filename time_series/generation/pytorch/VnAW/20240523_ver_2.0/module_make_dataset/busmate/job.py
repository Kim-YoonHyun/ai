import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import json
import copy
import random

from share_module import sensorutils as ssum

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils as utm

def preprocessing(yesterday, whole_df, onoff_df, system_list, target_name, fuel_type, whole_y_list, norm, info):
    for system in system_list:
        system_name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if system_name == target_name:
            break

    whole_df = ssum.xmy_preprocess(
        mode=f'{fuel_type}_{system_name}',
        whole_df=whole_df, 
        col_list=y_list,
        yesterday=yesterday, 
        onoff_df=onoff_df
    )

    # 변수 업데이트
    whole_y_list.extend(y_list)
    norm.update(y_norm)
    info[f'{target_name} y list'] = y_list
    return whole_df, whole_y_list, norm, info


def job(info, yesterday, fuel_type, whole_df_dict, 
        x_list, mark_list, system_list, norm, 
        y2x, interval_sec, x_seg_sec, zero_speed_p, train_p, 
        save_path):
    
    whole_day_total_num = 0
    whole_day_train_num = 0
    # on/off 불러오기
    for bus_name, whole_df in whole_df_dict.items():
        print(bus_name)
        whole_y_list = []
        # =========================================================================
        # on/off 계산
        try:
            if 'diesel' in fuel_type:
                onoff_df = ssum.ECU_preprocessing(whole_df)
            else:
                onoff_df = copy.deepcopy(whole_df)
                valid_ary = whole_df['ADC_ACC'].values
                valid_ary = np.where(valid_ary != 1, 0, valid_ary)
                onoff_df['valid'] = valid_ary
        except IndexError:
            continue
        
        # =========================================================================
        try:
            print('x, ', end='')
            whole_df = ssum.xmy_preprocess(
                mode='x',
                whole_df=whole_df, 
                col_list=x_list, 
                yesterday=yesterday, 
                onoff_df=onoff_df, 
                col_max=1000,
                col_min=0
            )
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
        # =========================================================================
        try:
            print('mark, ', end='')
            whole_df = ssum.xmy_preprocess(
                mode='mark',
                whole_df=whole_df, 
                col_list=mark_list, 
                yesterday=yesterday, 
                onoff_df=onoff_df, 
                col_max=180,
                col_min=-180
            )
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
        
        # =========================================================================
        # 제네레이터 센서 결측치 처리
        try:
            print('제네레이터, ', end='')
            whole_df, whole_y_list, norm, info = preprocessing(
                yesterday=yesterday, 
                whole_df=whole_df,
                onoff_df=onoff_df, 
                system_list=system_list, 
                target_name='제네레이터', 
                fuel_type=fuel_type, 
                whole_y_list=whole_y_list, 
                norm=norm, 
                info=info
            )
            print('냉각수온도, ', end='')
            whole_df, whole_y_list, norm, info = preprocessing(
                yesterday, whole_df, onoff_df, system_list, '냉각수온도', fuel_type, whole_y_list, norm, info)
            if fuel_type == 'diesel_new':
                print('DPF, ', end='')
                whole_df, whole_y_list, norm, info = preprocessing(
                    yesterday, whole_df, onoff_df, system_list, 'DPF', fuel_type, whole_y_list, norm, info)
                print('SCR, ', end='')
                whole_df, whole_y_list, norm, info = preprocessing(
                    yesterday, whole_df, onoff_df, system_list, 'SCR', fuel_type, whole_y_list, norm, info)
            if fuel_type == 'CNG':
                print('연료탱크, ', end='')
                whole_df, whole_y_list, norm, info = preprocessing(
                    yesterday, whole_df, onoff_df, system_list, '연료탱크', fuel_type, whole_y_list, norm, info)
            print('전처리')

        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
        
        # =========================================================================
        try:
            # 전체 컬럼 리스트 생성
            whole_y_list = list(set(whole_y_list))
            whole_col_list = x_list + whole_y_list + mark_list
    
            # 진행
            result_df = whole_df[whole_col_list]
            result_df = result_df.dropna()
            max_interval = len(result_df) // interval_sec
            seg_sec = x_seg_sec
            
            seg_df_list = []
            for i in range(max_interval):
                seg_df = result_df.iloc[i*interval_sec:i*interval_sec+seg_sec, :]
                seg_df = seg_df.dropna()
                
                # 결측치가 존재하는 경우 제외
                if len(seg_df) != seg_sec:
                    continue
                
                # 마지막에 다다른 경우 제외
                if len(seg_df) < seg_sec:
                    break
                
                # 속도가 0인 경우가 너무 많으면 제외
                if fuel_type == 'diesel_new':
                    speed_ary = seg_df['VCU_VehicleSpeed_STD'].values
                else:
                    speed_ary = seg_df['ECU_VehicleSpeed_STD'].values
                zero_ary = speed_ary[np.where(speed_ary == 0)]
                if len(zero_ary) > x_seg_sec * zero_speed_p / 100:
                    continue
                    
                # 입력        
                seg_df_list.append(seg_df)
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
        # =========================================================================
        try:
            day_total_num = len(seg_df_list)
            day_train_num = round(day_total_num * train_p / 100)
            
            whole_day_total_num += day_total_num
            whole_day_train_num += day_train_num
            
            # 셔플
            random.shuffle(seg_df_list)
    
            # 저장한 데이터별로 진행
            for i, seg_df in enumerate(tqdm(seg_df_list)):
                
                if i+1 <= day_train_num:
                    tv = 'train'
                else:
                    tv = 'val'
                
                # 계통별로 진행        
                for system in system_list:
                    system_name = system['name']
                    y_list = system['y_list']
                
                    # 센서별 진행
                    for y in y_list:
                        whole_x_list = x_list + y_list
                        if not y2x:
                            whole_x_list.remove(y)
                        seg_x_df = seg_df[whole_x_list]
                        seg_y_df = seg_df[[y]]
                        seg_mark_df = seg_df[mark_list].iloc[:x_seg_sec, :]
                        
                        # x, y, mark 순으로 저장
                        for direc in ['x', 'mark', 'y']:
                            if direc == 'x':
                                save_df = seg_x_df.copy()
                            if direc == 'y':
                                save_df = seg_y_df.copy()
                            if direc == 'mark':
                                save_df = seg_mark_df.copy()
                            
                            # 저장    
                            order = str(i).zfill(5)
                            xym_save_path = f'{save_path}/{system_name}/{y}/{tv}/{direc}'
                            os.makedirs(xym_save_path, exist_ok=True)
                            # save_df.to_csv(f'{xym_save_path}/{bus_name}_{yesterday}_{order}_{direc}.csv', index=False, encoding='utf-8-sig')
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
    # =========================================================================
    try:
        # norm 저장
        print('11111111111')
        if os.path.isfile(f'{save_path}/norm.json'):
            with open(f'{save_path}/norm.json', 'r', encoding='utf-8-sig') as f:
                pre_norm = json.load(f)
            norm.update(pre_norm)
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/norm.json', 'w', encoding='utf-8-sig') as f:
            json.dump(norm, f, indent='\t', ensure_ascii=False)
        
        # info 
        # key 순서 정하기
        info['train_p'] = train_p
        info['total_num'] = 0
        info['train_num'] = 0
        info['bus_name_list'] = list(whole_df_dict.keys())
        info['day'] = {}
        
        if os.path.isfile(f'{save_path}/info.json'):
            with open(f'{save_path}/info.json', 'r', encoding='utf-8-sig') as f:
                info = json.load(f)
            info['bus_name_list'].extend(list(whole_df_dict.keys()))
            info['bus_name_list'] = list(set(info['bus_name_list']))
            
        # 해당 날짜 갯수 입력
        info['day'][yesterday] = {
            'total_num':whole_day_total_num,
            'train_num':whole_day_train_num
        }
    
        # 전체 갯수 다시 리셋
        info['total_num'] = 0
        info['train_num'] = 0            
        
        # 전체 갯수 계산 
        day_num_dict = info['day']
        for num_dict in list(day_num_dict.values()):
            info['total_num'] += num_dict['total_num']
            info['train_num'] += num_dict['train_num']
        
        info['day'] = dict(sorted(info['day'].items(), 
                            key=lambda item: item[0],
                            reverse=False))
        
        # info 저장
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/info.json', 'w', encoding='utf-8-sig') as f:
            json.dump(info, f, indent='\t', ensure_ascii=False)
    except Exception:
        error_info = utm.get_error_info()
        print(error_info)
        raise ValueError()