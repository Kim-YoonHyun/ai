import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import json
import random
from share_module import sensorutils as ssum


def temp(yesterday, system_list, subject, whole_df_dict, onoff_df_dict, fuel_type, y_dict, norm, info):
    for system in system_list:
        name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if name == subject:
            break
    
    # 센서 전처리
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode=f'{fuel_type}_{name}',
            whole_df=whole_df, 
            col_list=y_list,
            yesterday=yesterday, 
            onoff_df=onoff_df
        )
        # 다시 저장
        whole_df_dict[bus_id] = whole_df
        
    del bus_id
    # dict 입력
    y_dict[name] = y_list
    
    # norm 업데이트
    norm.update(y_norm)
    
    # info
    info['DPF y list'] = y_list
    
    return whole_df_dict, y_dict, norm, info



def job(info, yesterday, fuel_type,
        whole_df_dict, x_list, mark_list, system_list, norm, 
        bos_num, y2x, interval_sec, x_seg_sec, zero_speed_p, train_p, 
        save_path):
    
    # =========================================================================
    # on/off 불러오기
    print('on/off 계산...')
    new_except_list = []
    onoff_df_dict = {}
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        try:
            # onoff_df = ssum.ECU_preprocessing(whole_df)
            pass
        except IndexError:
            new_except_list.append(bus_id)
            continue
        onoff_df_dict[bus_id] = onoff_df
    del bus_id

    # 추가 제회 항목 제거
    for new_except in new_except_list:
        whole_df_dict.pop(new_except)
        
    # =========================================================================
    print('x 데이터 전처리')
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode='x',
            whole_df=whole_df, 
            col_list=x_list, 
            yesterday=yesterday, 
            onoff_df=onoff_df, 
            col_max=1000,
            col_min=0
        )
        whole_df_dict[bus_id] = whole_df
    del bus_id
    
    # =========================================================================
    print('mark 데이터 전처리')
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode='mark',
            whole_df=whole_df, 
            col_list=mark_list, 
            yesterday=yesterday, 
            onoff_df=onoff_df, 
            col_max=180,
            col_min=-180
        )
        whole_df_dict[bus_id] = whole_df
    del bus_id
        
    # =========================================================================
    # 제네레이터 센서 결측치 처리
    print('제네레이터 센서 전처리...')
    y_dict = {}
    for system in system_list:
        name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if name == '제네레이터':
            break
    
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode=f'{fuel_type}_{name}',
            whole_df=whole_df, 
            col_list=y_list,
            yesterday=yesterday, 
            onoff_df=onoff_df
        )
        whole_df_dict[bus_id] = whole_df
        
    # 변수 업데이트
    y_dict[name] = y_list
    norm.update(y_norm)
    info['제네레이터 y list'] = y_list
    del bus_id, y_list, name, y_norm
        
    # =========================================================================
    # SCR 센서 결측치 처리
    print('냉각수온도 센서 전처리...')
    for system in system_list:
        name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if name == '냉각수온도':
            break
    
    # SCR 센서 전처리
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode=f'{fuel_type}_{name}',
            whole_df=whole_df, 
            col_list=y_list,
            yesterday=yesterday, 
            onoff_df=onoff_df
        )
        whole_df_dict[bus_id] = whole_df
    
    # 변수 업데이트
    y_dict[name] = y_list
    norm.update(y_norm)
    info['냉각수온도 y list'] = y_list
    del bus_id, y_list, y_norm, name
    
    # =========================================================================
    print('실린더실화 센서 전처리...')
    for system in system_list:
        name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if name == '실린더실화':
            break
    
    # Voltage 전처리
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode=f'{fuel_type}_{name}',
            whole_df=whole_df, 
            col_list=y_list,
            yesterday=yesterday, 
            onoff_df=onoff_df
        )
        whole_df_dict[bus_id] = whole_df
        
    # 변수 업데이트
    y_dict[name] = y_list
    norm.update(y_norm)
    info['실린더실화 y list'] = y_list
    del bus_id, y_list, name, y_norm
    
    # =========================================================================
    # 냉각수온도 센서 결측치 처리
    print('연료탱크 센서 전처리...')
    for system in system_list:
        name = system['name']
        y_list = system['y_list']
        y_norm = system['y_norm']
        if name == '연료탱크':
            break
    
    for bus_id, whole_df in tqdm(whole_df_dict.items()):
        onoff_df = onoff_df_dict[bus_id]
        whole_df = ssum.xmy_preprocess(
            mode=f'{fuel_type}_{name}',
            whole_df=whole_df, 
            col_list=y_list,
            yesterday=yesterday, 
            onoff_df=onoff_df
        )
        whole_df_dict[bus_id] = whole_df
        
    # 변수 업데이트
    y_dict[name] = y_list
    norm.update(y_norm)
    info['연료탱크 y list'] = y_list
    del bus_id, y_list, name, y_norm
        
    # =========================================================================
    print('데이터 분할...')
    whole_y_list = []
    for _, y_list in y_dict.items():
        whole_y_list.extend(y_list)
        
    # 전체 컬럼 리스트 생성
    whole_y_list = list(set(whole_y_list))
    whole_col_list = x_list + whole_y_list + mark_list
    
    # 분할
    seg_df_list = []
    for bus_id, whole_df in whole_df_dict.items():
        result_df = whole_df[whole_col_list]
        result_df = result_df.dropna()
    
        max_interval = len(result_df) // interval_sec
        seg_sec = x_seg_sec
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
            speed_ary = seg_df['VCU_VehicleSpeed_STD'].values
            zero_ary = speed_ary[np.where(speed_ary == 0)]
            if len(zero_ary) > x_seg_sec * zero_speed_p / 100:
                continue
                
            # 입력        
            seg_df_list.append(seg_df)
    del bus_id
    
    # =========================================================================
    print('학습 데이터 저장')
    # 학습데이터 비율 계산
    day_total_num = len(seg_df_list)
    day_train_num = round(day_total_num * train_p / 100)
    
    # norm 저장
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/norm.json', 'w', encoding='utf-8-sig') as f:
        json.dump(norm, f, indent='\t', ensure_ascii=False)
    
    # 셔플
    random.shuffle(seg_df_list)
    
    # 저장한 데이터별로 진행
    for i, seg_df in enumerate(tqdm(seg_df_list)):
        
        # bos 입력
        seg_df.iloc[:1, :] = bos_num
        
        # train & val 구분
        if i+1 <= day_train_num:
            tv = 'train'
        else:
            tv = 'val'
        
        # 계통별로 진행        
        for system in system_list:
            name = system['name']
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
                    xym_save_path = f'{save_path}/{name}/{y}/{tv}/{direc}'
                    os.makedirs(xym_save_path, exist_ok=True)
                    save_df.to_csv(f'{xym_save_path}/{yesterday}_{order}_{direc}.csv', index=False, encoding='utf-8-sig')
                    
    # =========================================================================
    # info 
    # key 순서 정하기
    info['train p'] = train_p
    info['total num'] = 0
    info['train num'] = 0
    info['bus id list'] = list(whole_df_dict.keys())
    info['day'] = {}
    
    # 만약 info 파일이 존재하는 경우
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