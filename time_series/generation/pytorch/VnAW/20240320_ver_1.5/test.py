import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import argparse

import torch

# from model_origin import network as net

# local modules
from mylocalmodules import dataloader as dam

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
# from sharemodule import classificationutils as clm
from sharemodule import utils as utm


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
    dummy_label = args.dummy_label
    test_data = args.test_data
    result_save_path = args.result_save_path
    
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
    
    try:
        device_num = int(device_num)
    except ValueError:
        device_num = args_setting['device_num']
    batch_size = args_setting['batch_size']
    shuffle = args_setting['shuffle']
    drop_last = args_setting['drop_last']
    num_workers = args_setting['num_workers']
    pin_memory = args_setting['pin_memory']
    random_seed = args_setting['random_seed']
    
    n_heads = args_setting['n_heads']
    x_len = args_setting['x_len']
    y_len = args_setting['y_len']
    sampler_name = args_setting['sampler_name']
    
    embed_type = args_setting['embed_type']
    d_model = args_setting['d_model']
    d_ff = args_setting['d_ff']
    temporal_type = args_setting['temporal_type']
    enc_layer_num = args_setting['enc_layer_num']
    enc_activation = args_setting['enc_activation']
    dec_layer_num = args_setting['dec_layer_num']
    dec_activation = args_setting['dec_activation']
    dropout_p = args_setting['dropout_p']
    network_name = args_setting['network_name']
    
    
    
    
    enc_feature_len = args_setting['enc_feature_len']
    enc_temporal_len = args_setting['enc_temporal_len']
    dec_feature_len = args_setting['dec_feature_len']
    dec_temporal_len = args_setting['dec_temporal_len']
    out_feature_len = args_setting['out_feature_len']
        
    
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # dataset 불러오기
    
    # =========================================================================
    print('get dataloader')
    dataset_path = f'{root_path}/datasets/{test_data}'
    
    # x val
    x_val_name_list = os.listdir(f'{dataset_path}/val/x')
    x_val_name_list.sort()
    
    # y val
    y_val_name_list = os.listdir(f'{dataset_path}/val/y')
    y_val_name_list.sort()
    
    # mark val
    mark_val_name_list = os.listdir(f'{dataset_path}/val/mark')
    mark_val_name_list.sort()
    
    # data 불러오기
    print('data loading...')
    
    # x val
    x_val_df_list = []
    for x_vn in x_val_name_list:
        x_val_df = pd.read_csv(f'{dataset_path}/val/x/{x_vn}')
        x_val_df_list.append(x_val_df)
        
    # y val
    true_val_df_list = []
    y_val_df_list = []
    for y_vn in y_val_name_list:
        true_val_df = pd.read_csv(f'{dataset_path}/val/y/{y_vn}')
        true_val_df_list.append(true_val_df.copy())
        true_val_df.iloc[1:, :] = -100
        y_val_df_list.append(true_val_df)
        
    # mark val
    mark_val_df_list = []
    for mvn in mark_val_name_list:
        mark_val_df = pd.read_csv(f'{dataset_path}/val/mark/{mvn}')
        mark_val_df_list.append(mark_val_df)
    
    # =========================================================================
    # dataloader 생성
    print('test dataloader 생성 중...')
    Test_Dataloader = dam.get_dataloader(
        x_df_list=x_val_df_list,
        x_mark_df_list=copy.deepcopy(mark_val_df_list),
        y_df_list=y_val_df_list,
        y_mark_df_list=copy.deepcopy(mark_val_df_list),
        n_heads=n_heads,
        pred_len=x_len,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler_name=sampler_name
    )
    # ===================================================================    
    print('get device')
    device = tum.get_device(device_num)
    
    if network_name == 'origin':
        from model_origin import network as net
        from model_origin import warm_up as wum
    if network_name == 'encoder_linear':
        from model_encoder_linear import network as net
        from model_encoder_linear import warm_up as wum
    # if network_name == 'linear':
    #     from model_linear import network as net
    #     from model_linear import warm_up as wum
    if network_name == 'lffnn':
        from model_lffnn import network as net
        from model_lffnn import warm_up as wum
    if network_name == 'res_lffnn':
        from model_res_lffnn import network as net
        from model_res_lffnn import warm_up as wum
        
    print('get_model')
    model = net.VnAW(
        x_len=x_len,
        y_len=y_len,
        embed_type=embed_type,
        d_model=d_model, 
        d_ff=d_ff,
        n_heads=n_heads, 
        temporal_type=temporal_type, 
        encoder_layer_num=enc_layer_num,
        encoder_feature_length=enc_feature_len, 
        encoder_temporal_length=enc_temporal_len,
        encoder_activation=enc_activation,
        decoder_layer_num=dec_layer_num,
        decoder_feature_length=dec_feature_len,
        decoder_temporal_length=dec_temporal_len,
        decoder_activation=dec_activation,
        dropout_p=dropout_p,
        output_length=out_feature_len
    )
    
    # 학습된 가중치 로딩
    weight_path = f'{trained_model_path}/{trained_model}/weight.pt'
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    logit_ary, true_ary, _, _ = trm.model_test(
        model=model, 
        test_dataloader=Test_Dataloader, 
        device=device
    )
    
    # ==
    from sharemodule import plotutils as plm
    
    n = 1
    
    # origin
    pred_y = np.squeeze(logit_ary[0])
    true_y = true_val_df_list[0].iloc[:, 0].values
    
    # lffnn
    # pred_y = np.squeeze(logit_ary[0])
    # true_y = true_val_df_list[0].iloc[:, 0].values
    
    print(pred_y)
    print(true_y)
    plm.draw_plot(
        title=test_data.split('/')[-1],
        x=range(1200),
        y=pred_y,
        add_x_list=[range(1200)],
        add_y_list=[true_y],
        fig_size=(20, 5),
        save_path=trained_model_path
    )
    # ==
    sys.exit()
    
    

        
if __name__  == '__main__':
    main()
