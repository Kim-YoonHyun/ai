import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import json
import random

from share_module import sensorutils as ssum

sys.path.append('/home/kimyh/python/ai')
from sharemodule import utils as utm


def job(info, yesterday, fuel_type,
        whole_df_dict, x_list, mark_list, system_list, norm, 
        bos_num, y2x, interval_sec, x_seg_sec, zero_speed_p, train_p, 
        save_path):
    
    # on/off 불러오기
    for bus_name, whole_df in whole_df_dict.items():
        print(bus_name)
        # =========================================================================
        # on/off 계산
        try:
            onoff_df = ssum.ECU_preprocessing(whole_df)
        except IndexError:
            continue
        
        # =========================================================================
        try:
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
        # DPF 센서 결측치 처리
        try:
            whole_y_list = []
            if fuel_type == 'diesel_new':
                for system in system_list:
                    system_name = system['name']
                    y_list = system['y_list']
                    y_norm = system['y_norm']
                    if system_name == 'DPF':
                        break
                
                # dpf 전처리
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
                info['DPF y list'] = y_list
                # del bus_name, y_list, name, y_norm
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
        
        # =========================================================================
        # SCR 센서 결측치 처리
        try:
            if fuel_type == 'diesel_new':
                for system in system_list:
                    system_name = system['name']
                    y_list = system['y_list']
                    y_norm = system['y_norm']
                    if system_name == 'SCR':
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
                info['SCR y list'] = y_list
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
        # =========================================================================
        try:
            for system in system_list:
                system_name = system['name']
                y_list = system['y_list']
                y_norm = system['y_norm']
                if system_name == '제네레이터':
                    break
    
            whole_df = ssum.xmy_preprocess(
                mode=f'{fuel_type}_{system_name}',
                whole_df=whole_df, 
                col_list=y_list,
                yesterday=yesterday, 
                onoff_df=onoff_df
            )
            # whole_df_dict[bus_id] = whole_df
        
            # 변수 업데이트
            whole_y_list.extend(y_list)
            norm.update(y_norm)
            info['Voltage y list'] = y_list
            # del bus_id, y_list, name, y_norm
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
        # =========================================================================
        # 냉각수온도 센서 결측치 처리
        try:
            for system in system_list:
                system_name = system['name']
                y_list = system['y_list']
                y_norm = system['y_norm']
                if system_name == '냉각수온도':
                    break
    
            whole_df = ssum.xmy_preprocess(
                mode=f'{fuel_type}_{system_name}',
                whole_df=whole_df, 
                col_list=y_list,
                yesterday=yesterday, 
                onoff_df=onoff_df
            )
            # whole_df_dict[bus_id] = whole_df
        
            # 변수 업데이트
            whole_y_list.extend(y_list)
            norm.update(y_norm)
            info['냉각수온도 y list'] = y_list
            # del bus_id, y_list, name, y_norm
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
            
            # 셔플
            random.shuffle(seg_df_list)
    

    
            # 저장한 데이터별로 진행
            for i, seg_df in enumerate(tqdm(seg_df_list)):
                
                # bos 입력
                if bos_num is not None:
                    seg_df.iloc[:1, :] = bos_num
                
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
                            save_df.to_csv(f'{xym_save_path}/{bus_name}_{yesterday}_{order}_{direc}.csv', index=False, encoding='utf-8-sig')
        except Exception:
            error_info = utm.get_error_info()
            print(error_info)
            continue
    
    # =========================================================================
    try:
        # norm 저장
        if os.path.isfile(f'{save_path}/norm.json'):
            with open(f'{save_path}/norm.json', 'r', encoding='utf-8-sig') as f:
                pre_norm = json.load(f)
            norm.update(pre_norm)
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/norm.json', 'w', encoding='utf-8-sig') as f:
            json.dump(norm, f, indent='\t', ensure_ascii=False)
        
        # info 
        # key 순서 정하기
        info['train p'] = train_p
        info['total num'] = 0
        info['train num'] = 0
        info['bus id list'] = list(whole_df_dict.keys())
        info['day'] = {}
        
        if os.path.isfile(f'{save_path}/info.json'):
            with open(f'{save_path}/info.json', 'r', encoding='utf-8-sig') as f:
                info = json.load(f)
            info['bus id list'].extend(list(whole_df_dict.keys()))
            info['bus id list'] = list(set(info['bus id list']))
            
        # 해당 날짜 갯수 입력
        info['day'][yesterday] = {
            'total num':day_total_num,
            'train num':day_train_num
        }
    
        # 전체 갯수 다시 리셋
        info['total num'] = 0
        info['train num'] = 0            
        
        # 전체 갯수 계산 
        day_num_dict = info['day']
        for num_dict in list(day_num_dict.values()):
            temp_total_n = num_dict['total num']
            temp_train_n = num_dict['train num']
            info['total num'] += temp_total_n
            info['train num'] += temp_train_n
        
        info['day'] = dict(sorted(info['day'].items(), 
                            key=lambda item: item[0],
                            reverse=False))
        
        # info 저장
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/info.json', 'w', encoding='utf-8-sig') as f:
            json.dump(info, f, indent='\t', ensure_ascii=False)
    except Exception:
        raise ValueError()