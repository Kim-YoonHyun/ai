





import sys
import os
sys.path.append(os.getcwd())
import json
import copy
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch

from mylocalmodules import dataloader as dam
from model_origin_3 import network as net
import model_trainer as mt

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')
    parser.add_argument('--device_num')
    parser.add_argument('--dummy_label', type=bool)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--result_save_path', type=str)
    args = parser.parse_args()
    return args


def device_num_setting(test_args_setting, train_args_setting):
    try:
        train_args_setting['device_num'] = int(test_args_setting['device_num'])
    except ValueError:
        pass
    return train_args_setting
    
    
# def make_ary_list(test_data, val_idx_ary, xym):
    
#     ary_list = []
#     name_list = os.listdir(f'{test_data}/val/{xym}')
#     name_list.sort()
#     filter_name_list = np.array(name_list)[val_idx_ary].tolist()
#     for filter_name in filter_name_list:
#         df = pd.read_csv(f'{test_data}/val/{xym}/{filter_name}', encoding='utf-8-sig')
#         ary = df.to_numpy()
#         ary_list.append(ary)

#     return ary_list


def tester(test_args_setting):
    # ==========================================================
    # 모델에 적용된 변수 불러오기
    print('변수 불러오기')
    with open(f"{test_args_setting['trained_model_path']}/args_setting.json", 'r', encoding='utf-8-sig') as f:
        train_args_setting = json.load(f)
    
    train_args_setting = device_num_setting(
        test_args_setting=test_args_setting,
        train_args_setting=train_args_setting
    )
    
    utm.envs_setting(train_args_setting['random_seed'])
    
    # =========================================================================
    # dataset 불러오기
    name_dict = {'val':{}}
    for xym in ['x', 'y', 'mark']:
        name_list = os.listdir(f"{test_args_setting['test_data']}/val/{xym}")
        name_list.sort()
        total_num = len(name_list)
        filter_name_list = mt.reducing(
            data_list=name_list,
            total_num=total_num,
            reduce_num=1
        )
        name_dict['val'][xym] = filter_name_list
        
    # norm 불러오기
    with open(f'{test_args_setting["test_data"]}/norm.json', 'r', encoding='utf-8-sig') as f:
        norm = json.load(f)
    
    # =========================================================================
    # dataloader 생성
    print('test dataloader 생성 중...')
    Test_Dataloader = dam.get_dataloader(
        mode='test',
        n_heads=train_args_setting['n_heads'],
        x_p=train_args_setting['x_p'],
        batch_size=train_args_setting['batch_size'], 
        dataset_path=f"{test_args_setting['test_data']}/val", 
        x_name_list=name_dict['val']['x'], 
        y_name_list=name_dict['val']['y'], 
        mark_name_list=name_dict['val']['mark'], 
        scale=train_args_setting['scale'],
        norm=norm,
        configuration=train_args_setting['configuration'],
        shuffle=train_args_setting['shuffle'], 
        drop_last=train_args_setting['drop_last'], 
        num_workers=train_args_setting['num_workers'], 
        pin_memory=train_args_setting['pin_memory'], 
        sampler_name=train_args_setting['sampler_name']
    )
    
    # true list 생성
    true_ary_list = []
    for y_name in name_dict['val']['y']:
        df = pd.read_csv(f"{test_args_setting['test_data']}/val/y/{y_name}", encoding='utf-8-sig')
        true_ary = df.to_numpy()
        true_ary_list.append(true_ary)
    
    # ===================================================================    
    print('get device')
    device = tum.get_device(train_args_setting['device_num'])
    
    print('get_model')
    model = net.VnAW(
        x_len=train_args_setting['x_len'], 
        y_len=train_args_setting['y_len'],
        embed_type=train_args_setting['embed_type'], 
        d_model=train_args_setting['d_model'], 
        d_ff=train_args_setting['d_ff'], 
        n_heads=train_args_setting['n_heads'], 
        projection_type=train_args_setting['projection_type'],
        temporal_type=train_args_setting['temporal_type'], 
        enc_layer_num=train_args_setting['enc_layer_num'],
        dec_layer_num=train_args_setting['dec_layer_num'],
        enc_feature_len=train_args_setting['enc_feature_len'], 
        dec_feature_len=train_args_setting['dec_feature_len'],
        enc_temporal_len=train_args_setting['enc_temporal_len'],
        dec_temporal_len=train_args_setting['dec_temporal_len'],
        enc_act=train_args_setting['enc_activation'],
        dec_act=train_args_setting['dec_activation'],
        dropout_p=train_args_setting['dropout_p'],
        output_length=train_args_setting['out_feature_len']
    )
    
    # 학습된 가중치 로딩
    weight_path = f"{test_args_setting['trained_model_path']}/{test_args_setting['trained_model']}/weight.pt"
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    
    # test 진행
    logit_ary, _, _, _ = trm.model_test(
        model=model, 
        test_dataloader=Test_Dataloader, 
        device=device
    )
    
    # ==
    from sharemodule import plotutils as plm
    
    # origin
    # sensor_name = test_args_setting["test_data"].split("/")[-1]
    sensor_norm_num = norm['ECU_DOCGasTemperature_After']

    true_y = np.squeeze(true_ary_list[0])
    pred_y = np.squeeze(logit_ary[0])[-len(true_y):] * sensor_norm_num
    
    if train_args_setting['configuration'] == 'bos time series':
        pred_y[:-600] = np.nan
    
    result_df = pd.DataFrame([pred_y, true_y], index=['pred', 'true']).T
    os.makedirs(test_args_setting["result_save_path"], exist_ok=True)
    result_df.to_csv(f'{test_args_setting["result_save_path"]}/result_df.csv', index=False, encoding='utf-8-sig')
    
    plm.draw_plot(
        title=f'result_{test_args_setting["trained_model"]}',
        x=range(len(true_y)),
        y=pred_y,
        title_font_size=20,
        x_font_size=20,
        y_font_size=20,
        line_color='black',
        add_x_list=[range(len(true_y))],
        add_y_list=[true_y],
        fig_size=(20, 5),
        save_path=test_args_setting['trained_model_path']
    )
    # ==
    sys.exit()
    
def main():
    args = get_args()    
    test_args_setting = vars(args)
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TEST',
        root_path=test_args_setting['root_path'],
        log_file_name=f'test.log',
        time_handler=True
    )
    
    # =========================================================================
    # 평가 진행
    tester(test_args_setting)

        
if __name__  == '__main__':
    main()
