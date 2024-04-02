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
    # enc_temporal_len = args_setting['enc_temporal_len']
    dec_feature_len = args_setting['dec_feature_len']
    # dec_temporal_len = args_setting['dec_temporal_len']
    out_feature_len = args_setting['out_feature_len']
        
    
    utm.envs_setting(random_seed)
    
    # =========================================================================
    # dataset 불러오기
    
    # =========================================================================
    # 데이터 불러오기
    print('get dataloader')
    dataset_path = f'{root_path}/datasets/{test_data}'
    
    # ==
    # 갯수 줄이기
    # train_num = len(os.listdir(f'{dataset_path}/train/x'))
    val_num = len(os.listdir(f'{dataset_path}/val/x'))
    # train_idx_ary = np.random.choice(range(train_num), size=20000, replace=False)
    val_idx_ary = np.random.choice(range(val_num), size=1, replace=False)
    # ==
    
    data_dict = {'val':{}}
    for xym in ['x', 'y']:#, 'mark']:
        name_list = os.listdir(f'{dataset_path}/val/{xym}')
        name_list.sort()
        filter_name_list = np.array(name_list)[val_idx_ary].tolist()
        
        ary_list = []
        print('val', xym)
        for name in tqdm(filter_name_list):
            df = pd.read_csv(f'{dataset_path}/val/{xym}/{name}')
            ary = df.to_numpy()
            ary_list.append(ary)
        data_dict['val'][xym] = ary_list
    x_val_ary_list = data_dict['val']['x']
    true_val_ary_list = data_dict['val']['y']
    # mark_val_df_list = data_dict['val']['mark']
    
    y_val_ary_list = []
    for true_val_ary in true_val_ary_list:
        temp_ary = true_val_ary.copy()
        # true_val_df_list.append(true_val_df.copy())
        temp_ary[1:, :] = -100
        y_val_ary_list.append(temp_ary)
    
    # # ==
    # true_val_ary_list = [true_val_ary_list[0]]
    # x_val_ary_list = [x_val_df_list[0]]
    # y_val_ary_list = [y_val_df_list[0]]
    # mark_val_ary_list = [mark_val_df_list[0]]
    # # ==
    
    # =========================================================================
    # 마스크 생성
    print('마스크 생성 중...')
    # val_self_mask_list = []
    val_la_mask_list = []
    mask_dict = {'val':{}}
    print('val')
    x_tv_ary_list = data_dict['val']['x']
    y_tv_ary_list = data_dict['val']['y']
    
    # self_mask_list = []
    la_mask_list = []
    for x_ary, y_ary in zip(x_tv_ary_list, tqdm(y_tv_ary_list)):
        # x_ = x_df.iloc[:, :1].values
        # x_ = np.squeeze(x_) 
        # self_mask, _ = dam.get_self_mask(x_, pred_len=x_len, n_heads=n_heads)
        # self_mask_list.append(self_mask)
        # y_ = y_ary.iloc[:, :1].values
        y_ = np.squeeze(y_ary)
        _, look_ahead_mask = dam.get_self_mask(y_, pred_len=y_len, n_heads=n_heads)
        la_mask_list.append(look_ahead_mask)
        
    # mask_dict['val']['self_mask'] = self_mask_list
    mask_dict['val']['la_mask'] = la_mask_list
            
    # val_self_mask_list = mask_dict['val']['self_mask']
    val_la_mask_list = mask_dict['val']['la_mask']
    
    # =========================================================================
    # dataloader 생성
    print('test dataloader 생성 중...')
    # Test_Dataloader = dam.get_dataloader(
    #     x_df_list=x_val_df_list,
    #     y_df_list=y_val_df_list,
    #     x_mark_df_list=copy.deepcopy(mark_val_df_list),
    #     y_mark_df_list=copy.deepcopy(mark_val_df_list),
    #     batch_size=batch_size, 
    #     shuffle=shuffle, 
    #     drop_last=drop_last, 
    #     num_workers=num_workers, 
    #     pin_memory=pin_memory, 
    #     sampler_name=sampler_name
    # )
    
    Test_Dataloader = dam.get_dataloader(
        x_list=x_val_ary_list,
        y_list=y_val_ary_list,
        # x_mark_df_list=copy.deepcopy(mark_val_df_list),
        # y_mark_df_list=copy.deepcopy(mark_val_df_list),
        # self_mask_list=val_self_mask_list,
        la_mask_list=val_la_mask_list,
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
    
    print('get_model')
    
    if network_name == 'origin':
        from model_origin import network as net
        from model_origin import warm_up as wum
    if network_name == 'decoder_linear':
        from model_decoder_linear import network as net
        model = net.VnAW(
            x_len=x_len, 
            y_len=y_len,
            embed_type=embed_type,
            d_model=d_model, 
            d_ff=d_ff,
            n_heads=n_heads, 
            encoder_feature_length=enc_feature_len, 
            decoder_layer_num=dec_layer_num,
            decoder_feature_length=dec_feature_len,
            decoder_activation=dec_activation,
            dropout_p=dropout_p,
            output_length=out_feature_len
        )
    # if network_name == 'linear':
    #     from model_linear import network as net
    #     from model_linear import warm_up as wum
    if network_name == 'lffnn':
        from model_lffnn import network as net
        from model_lffnn import warm_up as wum
    if network_name == 'res_lffnn':
        from model_res_lffnn import network as net
        from model_res_lffnn import warm_up as wum
        
    
    # model = net.VnAW(
    #     x_len=x_len,
    #     y_len=y_len,
    #     embed_type=embed_type,
    #     d_model=d_model, 
    #     d_ff=d_ff,
    #     n_heads=n_heads, 
    #     temporal_type=temporal_type, 
    #     encoder_layer_num=enc_layer_num,
    #     encoder_feature_length=enc_feature_len, 
    #     encoder_temporal_length=enc_temporal_len,
    #     encoder_activation=enc_activation,
    #     decoder_layer_num=dec_layer_num,
    #     decoder_feature_length=dec_feature_len,
    #     decoder_temporal_length=dec_temporal_len,
    #     decoder_activation=dec_activation,
    #     dropout_p=dropout_p,
    #     output_length=out_feature_len
    # )
    
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
    true_y = np.squeeze(true_val_ary_list[0])
    
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
