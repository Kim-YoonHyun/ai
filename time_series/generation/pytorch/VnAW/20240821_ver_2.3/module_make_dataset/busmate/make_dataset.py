import sys
import os
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import json
from datetime import date, datetime, timedelta

# import job_diesel as jd
# import job_cng as jc
import job as j

sys.path.append(os.getcwd())
# from share_module import timeutils as tim
# from share_module import sensorutils as ssum

sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import utils as utm



develop = True
if develop:
    import develop_config as config
else:
    import config


def start_job(yesterday, config, log):
    info = {
        'dataset_name':config.new_dataset_name,
        'fuel_type':config.fuel_type,
        'configuration':config.configuration,
        'data_interval':config.data_interval,
        'x_size':config.x_size,
        'y_size':config.y_size,
        'x_feature_length':None,
        'y_feature_length':None,
        'x_list':None,
        'y_list':None,
        'column_json_name':config.column_json_name,
        'valid_json_name':config.valid_json_name,
        'train_p':config.train_p,
        'count':{
            'total':{
                'total':0,
                'train':0,
                'val':0
            },
            'each':{}
        },
        'bus_name_list':None
    }
    
    # =========================================================================
    status, error_info, error_msg = j.job(
        config=config,
        info=info,
        yesterday=yesterday,
    )
    return status, error_info, error_msg
    


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
    date_list = utm.get_date_list(
        schedule=False,
        year=config.year,
        mon_list=config.mon_list,
        start_day_list=config.start_day_list,
        end_day_list=config.end_day_list
    )
        
    # 실행
    for yesterday in date_list:
        status, error_info, error_msg = start_job(yesterday, config, log)
        
        
if __name__ == '__main__':
    main()