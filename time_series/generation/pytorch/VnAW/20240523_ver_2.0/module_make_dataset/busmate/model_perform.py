import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import json
import torch

sys.path.append('/home/kimyh/python/ai/time_series/generation/pytorch/VnAW')
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
from sharemodule import plotutils as plm


import develop_config as config


def get_score(pred, true, constance=2, save_path=None, file_name=None):
    # MSE
    mse = np.sqrt(np.square(pred - true))
    mse_aver = np.average(mse)
    
    # std
    std = np.std(mse)
    
    # IQR
    q1 = np.percentile(mse, 25)
    q3 = np.percentile(mse, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr*constance
    higher_bound = q3 + iqr*constance

    # worst
    w_mse1 = mse[np.where(mse < lower_bound)]
    w_mse2 = mse[np.where(mse > higher_bound)]
    if len(w_mse1) == 0:
        worst_aver1 = 0
    else:
        worst_aver1 = np.average(w_mse1)
    if len(w_mse2) == 0:
        worst_aver2 = 0
    else:
        worst_aver2 = np.average(w_mse2)
    worst_aver = worst_aver1 + worst_aver2

    # 
    excluded_mse = mse[np.where(mse >= lower_bound)]
    excluded_mse = mse[np.where(excluded_mse <= higher_bound)]
    
    anormal_count = len(mse) - len(excluded_mse)
    exclu_aver = np.average(excluded_mse)

    score = int(exclu_aver * worst_aver * std / 1000)

    if save_path is not None:
        score_dict = {
            'score':int(score),
            '이상치 갯수':int(anormal_count),
            'total aver':int(mse_aver),
            'exclu aver':int(exclu_aver),
            'worst aver':int(worst_aver),
            'std':int(std)
        }
        os.makedirs(save_path, exist_ok=True)
        if file_name is None:
            with open(f'{save_path}/score.json', 'w', encoding='utf-8-sig') as f:
                json.dump(score_dict, f, indent='\t', ensure_ascii=False)
        else:
            with open(f'{save_path}/{file_name}.json', 'w', encoding='utf-8-sig') as f:
                json.dump(score_dict, f, indent='\t', ensure_ascii=False)
    return score, exclu_aver, worst_aver, std


def main():
    root_path = config.root_path
    phase = 'phase_2nd'
    dataset = 'dataset_01'
    condition_order = '0000'
    
    fuel_type_path = f'{root_path}/trained_model/{phase}/{dataset}'
    fuel_type_list = os.listdir(fuel_type_path)
    
    try:
        with open(f'./image_{phase}_{dataset}/score_info.json', 'r', encoding='utf-8-sig') as f:
            score_info = json.load(f)
    except FileNotFoundError:
        score_info = {
            "diesel":{
                "제네레이터":{
                    "ECU_BatteryVoltage":{}, 
                    "VCU_EngineSpeed":{},
                },
                "냉각수온도":{
    				"VCU_CoolantTemperature":{},
                    "VCU_EngineSpeed":{}
                },
				"DPF":{
				    "ECU_DOCGasTemperature_Before":{},
                    "ECU_DOCGasTemperature_After":{}
				},
				"SCR":{
				    "SCR_CatalystTemperature_Before":{}, 
                    "SCR_DosingModuleDuty":{}, 
                    "SCR_CatalystNOx_Before":{}, 
                    "SCR_CatalystNOx_After":{}
                }
            },
    		"CNG":{
				"제네레이터":{
				    "ECU_BatteryVoltage":{}, 
                    "ECU_EngineSpeed":{}
                },
				"냉각수온도":{
				    "ECU_CoolantTemperature":{}, 
                    "ECU_EngineSpeed":{}
                },
				"연료탱크":{
				"ECU_FuelTankPressure":{}
		    	}
            }
        }
    for fuel_type in fuel_type_list:
        system_path = f'{fuel_type_path}/{fuel_type}'
        system_list = os.listdir(system_path)
        
        for system in system_list:
            sensor_path = f'{system_path}/{system}'
            sensor_list = os.listdir(sensor_path)
            
            #==
            if system != 'DPF':
                continue
            #==
            for sensor in sensor_list:
                #==
                if sensor != 'ECU_DOCGasTemperature_Before':
                    continue
                #==
                try:
                    _ = os.listdir(f'{sensor_path}/{sensor}/{condition_order}')
                    epoch_path = f'{sensor_path}/{sensor}/{condition_order}'
                except FileNotFoundError:
                    continue
                
                epoch = 1
                while True:
                    try:
                        _ = os.listdir(f'{epoch_path}/epoch{str(epoch).zfill(4)}')
                        weight_path = f'{epoch_path}/epoch{str(epoch).zfill(4)}/weight.pt'
                        break
                    except Exception:
                        epoch += 1
                print(weight_path)
                # =======================================================================
                val_path = f'/data/busmate/datasets/{dataset}/{fuel_type}/{system}/{sensor}/val'
                
                print('변수 불러오기')
                with open(f'{epoch_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
                    args_setting = json.load(f)
                
                scale = args_setting['scale']
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
                x_len = args_setting['x_len']
                y_len = args_setting['y_len']
                n_heads = args_setting['n_heads']
                embed_type = args_setting['embed_type']
                projection_type = args_setting['projection_type']
                temporal_type = args_setting['temporal_type']
                
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
                
                
                
                
                # norm 불러오기
                norm_path = '/'.join(val_path.split('/')[:-3])
                with open(f'{norm_path}/norm.json', 'r', encoding='utf-8-sig') as f:
                    norm = json.load(f)
                
                # ==
                # 갯수 줄이기
                val_num = len(os.listdir(f'{val_path}/x'))
                val_idx_ary = np.random.choice(range(val_num), size=1, replace=False)
                val_idx_ary = [0]
                # print(val_idx_ary)
                # sys.exit()
                # ==

                # 데이터 리스트 생성
                data_dict = {'val':{}}
                
                true_ary_list = []
                # x, y, mark 별로 진행
                for xym in ['x', 'y', 'mark']:
                    name_list = os.listdir(f'{val_path}/{xym}')
                    name_list.sort()
                    
                    filter_name_list = np.array(name_list)[val_idx_ary].tolist()
                    
                    temp_ary_list = []
                    for filter_name in filter_name_list:
                        temp_df = pd.read_csv(f'{val_path}/{xym}/{filter_name}', encoding='utf-8-sig')
                        
                        if scale == 'fixed-max':
                            temp_df = dam.fixed_max_norm(temp_df, norm)
                        temp_ary = temp_df.to_numpy()
                        
                        # y 인 경우            
                        if xym == 'y':
                            true_ary = temp_ary.copy()
                            true_ary_list.append(true_ary)
                            temp_ary = np.array([[3]])
                            
                        temp_ary_list.append(temp_ary)
                    data_dict['val'][xym] = temp_ary_list
        
                
                # dataloader 생성
                print('test dataloader 생성 중...')
                Test_Dataloader = dam.get_test_dataloader(
                    x_list=data_dict['val']['x'],
                    # y_list=data_dict['val']['y'],
                    y_list=true_ary_list,
                    true_list=true_ary_list,
                    n_heads=n_heads,
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    drop_last=drop_last, 
                    num_workers=num_workers, 
                    pin_memory=pin_memory, 
                    sampler_name=sampler_name
                )
                
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

                print(model)
                weight = torch.load(weight_path, map_location=device)
                model.load_state_dict(weight)
                model.to(device)
                logit_ary, _, _, _ = trm.model_test(
                    model=model, 
                    test_dataloader=Test_Dataloader, 
                    device=device
                )
                
                # ==
                
                
                n = 1
                # denormalize
                norm_num = norm[sensor]
                
                # origin
                pred_y = np.squeeze(logit_ary[0])[1:] * norm_num
                true_y = np.squeeze(true_ary_list[0])[1:] * norm_num
                
                score, exclu_aver, worst_aver, std = get_score(pred_y, true_y)
                

                score_info[fuel_type][system][sensor][condition_order] = {
                    'score':score, 
                    'exclu_aver':exclu_aver, 
                    'worst_aver':worst_aver, 
                    'std':std
                }

                # result_df = pd.DataFrame([pred_y, true_y], index=['pred', 'true']).T
                # print(result_df)
                # result_df.to_csv(f'{root_path}/result_df_{test_data.split("/")[-1]}.csv', index=False, encoding='utf-8-sig')
                # true_y = np.squeeze(true_ary[0])
                
                # lffnn
                # pred_y = np.squeeze(logit_ary[0])
                # true_y = true_val_df_list[0].iloc[:, 0].values
                
                plm.draw_plot(
                    title=f'{fuel_type}_{system}_{sensor}_{condition_order}_{str(epoch).zfill(4)}_{score}',
                    x=range(len(true_y)),
                    y=pred_y,
                    # y_range=(0, norm_num),
                    line_color='black',
                    add_x_list=[range(len(true_y))],
                    add_y_list=[true_y],
                    fig_size=(20, 5),
                    save_path='./image/'
                )
                # ==
                # sys.exit()
    with open(f'./image/score_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(score_info, f, indent='\t', ensure_ascii=False)



if __name__ == '__main__':
    main()
