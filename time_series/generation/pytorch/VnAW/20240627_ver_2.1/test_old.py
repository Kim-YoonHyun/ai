import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import argparse

import torch

from model_origin_3 import network as net

# local modules
from mylocalmodules import dataloader as dam

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
# from sharemodule import classificationutils as clm
from sharemodule import utils as utm


def make_ary_list(test_data, val_idx_ary, xym, scale, norm):
    ary_list = []
    name_list = os.listdir(f'{test_data}/val/{xym}')
    name_list.sort()
    filter_name_list = np.array(name_list)[val_idx_ary].tolist()
    for filter_name in filter_name_list:
        df = pd.read_csv(f'{test_data}/val/{xym}/{filter_name}', encoding='utf-8-sig')
        # temp_df = pd.read_csv(f'{test_data}/val/{xym}/{filter_name}', encoding='utf-8-sig')
            
        if scale == 'fixed-max':
            df = dam.fixed_max_norm(df, norm)
            
        ary = df.to_numpy()
        ary_list.append(ary)

    return ary_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model')
    parser.add_argument('--device_num')
    parser.add_argument('--dummy_label', type=bool)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--result_save_path', type=str)

    args = parser.parse_args()
    args_setting = vars(args)
    
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model = args.trained_model
    device_num = args.device_num
    test_data = args.test_data
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TEST',
        root_path=root_path,
        log_file_name=f'test.log',
        time_handler=True
    )
    
    # ==========================================================
    # 모델에 적용된 변수 불러오기
    print('변수 불러오기')
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        args_setting = json.load(f)
    
    scale = args_setting['scale']
    x_p = args_setting['x_p']
    try:
        device_num = int(device_num)
    except ValueError:
        device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    dropout_p = args_setting['dropout_p']
    random_seed = args_setting['random_seed']
    sampler_name = args_setting['sampler_name']
    shuffle = args_setting['shuffle']
    drop_last = args_setting['drop_last']
    num_workers = args_setting['num_workers']
    pin_memory = args_setting['pin_memory']
    
    # network parameter
    d_model = args_setting['d_model']
    d_ff = args_setting['d_ff']
    n_heads = args_setting['n_heads']
    embed_type = args_setting['embed_type']
    temporal_type = args_setting['temporal_type']
    projection_type = args_setting['projection_type']
    
    # layer num
    enc_layer_num = args_setting['enc_layer_num']
    dec_layer_num = args_setting['dec_layer_num']
    
    # activation
    enc_activation = args_setting['enc_activation']
    dec_activation = args_setting['dec_activation']
    

    enc_feature_len = args_setting['enc_feature_len']
    enc_temporal_len = args_setting['enc_temporal_len']
    dec_feature_len = args_setting['dec_feature_len']
    dec_temporal_len = args_setting['dec_temporal_len']
    out_feature_len = args_setting['out_feature_len']
        
    
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # dataset 불러오기
    print('get dataloader')
    # dataset_path = f'{root_path}/datasets/{test_data}'
    
    # norm 불러오기
    norm_path = '/'.join(test_data.split('/')[:-2])
    with open(f'{norm_path}/norm.json', 'r', encoding='utf-8-sig') as f:
        norm = json.load(f)
    
    # ==
    # 갯수 줄이기
    val_num = len(os.listdir(f'{test_data}/val/x'))
    val_idx_ary = np.random.choice(range(val_num), size=1, replace=False)
    # ==

    # 데이터 리스트 생성
    data_dict = {'val':{}}
    
    #==
    # x        
    x_ary_list = make_ary_list(
        test_data=test_data, 
        val_idx_ary=val_idx_ary, 
        xym='x', 
        scale=scale, 
        norm=norm
    )
    data_dict['val']['x'] = x_ary_list
    
    # true
    true_ary_list = make_ary_list(
        test_data=test_data, 
        val_idx_ary=val_idx_ary, 
        xym='y', 
        scale=scale, 
        norm=norm
    )
    
    # y
    if x_p > 0:
        y_ary_list = []
        for x_ary in x_ary_list:
            x_len = len(x_ary)
            zero_ary = np.zeros((x_len, 1))
            y_ary_list.append(zero_ary)
    else:
        y_ary_list = copy.deepcopy(true_ary_list)
    data_dict['val']['y'] = y_ary_list
    
    # mark
    mark_ary_list = make_ary_list(
        test_data=test_data, 
        val_idx_ary=val_idx_ary, 
        xym='mark', 
        scale=scale, 
        norm=norm
    )
    data_dict['val']['mark'] = mark_ary_list
    
    # =========================================================================
    # dataloader 생성
    print('test dataloader 생성 중...')
    Test_Dataloader = dam.get_dataloader(
        mode='test',
        n_heads=n_heads,
        x_p=x_p,
        batch_size=batch_size, 
        x_list=data_dict['val']['x'],
        y_list=data_dict['val']['y'],
        mark_list=data_dict['val']['mark'],
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler_name=sampler_name
    )
    temp_dataloader = copy.deepcopy(Test_Dataloader)
    x, mark, _, y, _, _, _ = next(iter(temp_dataloader))
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    mark = mark.cpu().detach().numpy()
    
    x_len = x.shape[1]
    y_len = y.shape[1]
    
    # ===================================================================    
    print('get device')
    device = tum.get_device(device_num)
    
    print('get_model')
    model = net.VnAW(
        x_len=x_len, 
        y_len=y_len,
        embed_type=embed_type, 
        d_model=d_model, 
        d_ff=d_ff, 
        n_heads=n_heads, 
        projection_type=projection_type,
        temporal_type=temporal_type, 
        enc_layer_num=enc_layer_num,
        dec_layer_num=dec_layer_num,
        enc_feature_len=enc_feature_len, 
        dec_feature_len=dec_feature_len,
        enc_temporal_len=enc_temporal_len,
        dec_temporal_len=dec_temporal_len,
        enc_act=enc_activation,
        dec_act=dec_activation,
        dropout_p=dropout_p,
        output_length=out_feature_len
    )
    
    # 학습된 가중치 로딩
    weight_path = f'{trained_model_path}/{trained_model}/weight.pt'
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    # ==    
    # weights = model.state_dict()
    # # 가중치 딕셔너리의 키와 값을 출력
    # for param_tensor in weights:
    #     print(param_tensor, "\t", weights[param_tensor].size())
    #     print(weights[param_tensor])
    # sys.exit()
    # ==
    
    logit_ary, _, _, _ = trm.model_test(
        model=model, 
        test_dataloader=Test_Dataloader, 
        device=device
    )
    
    # ==
    from sharemodule import plotutils as plm
    
    n = 1
    
    # origin
    pred_y = np.squeeze(logit_ary[0])[1:]
    true_y = np.squeeze(true_ary_list[0])[1:]
    
    result_df = pd.DataFrame([pred_y, true_y], index=['pred', 'true']).T
    # print(result_df)
    result_df.to_csv(f'{root_path}/result_df_{test_data.split("/")[-1]}.csv', index=False, encoding='utf-8-sig')
    # true_y = np.squeeze(true_ary[0])
    
    # lffnn
    # pred_y = np.squeeze(logit_ary[0])
    # true_y = true_val_df_list[0].iloc[:, 0].values
    
    plm.draw_plot(
        title=test_data.split('/')[-1],
        x=range(len(true_y)),
        y=pred_y,
        line_color='black',
        add_x_list=[range(len(true_y))],
        add_y_list=[true_y],
        fig_size=(20, 5),
        save_path=trained_model_path
    )
    # ==
    sys.exit()
    
    

        
if __name__  == '__main__':
    main()
