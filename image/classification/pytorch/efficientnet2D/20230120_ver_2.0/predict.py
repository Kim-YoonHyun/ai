'''
install requirement
!pip install timm
'''

# from datetime import datetime, timezone, timedelta
import random
import os
import sys
import copy
import numpy as np
import pandas as pd
import argparse
import timm
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json


from mylocalmodules import modelutils as mutils
from mylocalmodules import datasetutils as dutils
from mylocalmodules import trainutils as tutils
from mylocalmodules import utils as lutils

sys.path.append('/home/kimyh/python')
from myglobalmodules import utils as gutils

def main():
    
    # parsing
    parser = argparse.ArgumentParser()
    
    # path parsing
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('--network', default='effnet')
    
    # model parsing
    parser.add_argument('trained_model')
    parser.add_argument('--devcie_num', type=int, default=0)
    
    # data parsing
    parser.add_argument('dataset_name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=False)
    
    # envs parsing
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    # environment setting
    gutils.envs_setting(args.random_seed)

    # make logger
    logger = gutils.get_logger('train', logging_level='info')

    # class_dict
    class_list = ['1++', '1+', '1', '2', '3']
    class_dict = {'1++':0, '1+':1, '1':2, '2':3, '3':4}

    # annotation
    test_annotation_df = pd.read_csv(f'{args.root_path}/datasets/{args.dataset_name}/test/annotation.csv')

    # Dataset
    Test_Dataset = dutils.TestDataset(img_path=f'{args.root_path}/datasets/{args.dataset_name}/test/images',
                                        annotation_df=test_annotation_df)
                                        # image_correction=True,
                                        # high_hsv_dict=high_hsv_dict,
                                        # row_hsv_dict=row_hsv_dict,
                                        # gamma=1.8)

    # DataLoader
    test_dataloader = DataLoader(dataset=Test_Dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                pin_memory=args.pin_memory,
                                drop_last=args.drop_last)
    # GPU
    device = gutils.get_device(args.devcie_num)

    # Load model
    if args.network == 'effnet':
        model_args = {'n_outputs':len(class_list)}
    model = mutils.get_model(model_name=args.network,
                      model_args=model_args).to(device)
    
    # predict
    print('\n>>> predict...')
    trained_model = model
    trained_model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_model}/weight.pt'))
    trained_model.eval()
    with torch.no_grad():
        test_pred_label_ary, _ = tutils.get_output(
            mode='test',
            model=trained_model, 
            dataloader=test_dataloader,
            class_list=class_list,
            device=device
            )

    # result df save
    gutils.createfolder(f'{args.root_path}/{args.trained_model}/{args.dataset_name}_predict')
    best_test_pred_class_ary = []
    for pred_label in test_pred_label_ary:
        best_test_pred_class_ary.append(class_list[pred_label])
        
    result_df = pd.DataFrame([test_annotation_df['name'].values, best_test_pred_class_ary, ], index=['id', 'grade']).T
    gutils.save_csv(save_path=f'{args.root_path}/{args.trained_model}/{args.dataset_name}_predict/predict_result_df.csv', data_for_save=result_df, index=False)


if __name__ == '__main__':
    main()
            
