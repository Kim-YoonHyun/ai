'''
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install transformers
'''
import sys
import os
sys.path.append(os.getcwd())

import main

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import utils as utm


def main2():
    args = main.get_args()
    args_setting = vars(args)
    args_setting = main.temporal_type_setting(args, args_setting)
    args_setting = main.activation_setting(args, args_setting)
    
    # =========================================================================
    utm.envs_setting(args_setting['random_seed'])

    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN_continue',
        root_path=args_setting['root_path'],
        log_file_name='train_continue.log',
        time_handler=True
    )
    
    dataset_name_dict = {
        '0':[
            'dataset_01/diesel/DPF/ECU_DOCGasTemperature_Before',
            
        ],
        '1':[
            # 'dataset_01/diesel/SCR/SCR_CatalystTemperature_Before',
            # 'dataset_01/diesel/SCR/SCR_DosingModuleDuty',
            # 'dataset_01/diesel/냉각수온도/VCU_CoolantTemperature'
        ],
        '2':[
            'dataset_01/diesel/SCR/SCR_CatalystNOx_After',
            'dataset_01/diesel/SCR/SCR_CatalystNOx_Before'
            
            # 'dataset_01/diesel/냉각수온도/VCU_EngineSpeed',
            # 'dataset_01/diesel/제네레이터/ECU_BatteryVoltage'
        ],
        '3':[
            'dataset_01/diesel/DPF/ECU_DOCGasTemperature_After',
            # 'dataset_01/diesel/제네레이터/VCU_EngineSpeed',
            # 'dataset_01/CNG/냉각수온도/ECU_CoolantTemperature',
            # 'dataset_01/CNG/냉각수온도/ECU_EngineSpeed',
            # 'dataset_01/CNG/연료탱크/ECU_FuelTankPressure',
            # 'dataset_01/CNG/제네레이터/ECU_BatteryVoltage',
            # 'dataset_01/CNG/제네레이터/ECU_EngineSpeed'
        ]
    }
    dataset_name_list = dataset_name_dict[str(args_setting['device_num'])]
    
    
    for dataset_name in dataset_name_list:
        args_setting['dataset_name'] = dataset_name
        main.trainer(args_setting)
        
        
        
if __name__ == "__main__":
    main2()