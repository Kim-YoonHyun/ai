# common module
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
from PIL import Image
import cv2

# custom module
from mylocalmodules import dataloader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils

sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils


def main():
    # -------------------------------------------------------------------------
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing 
    parser.add_argument('root_path', help='프로젝트 디렉토리의 경로')
    parser.add_argument('for_test_dataset_name')
    parser.add_argument('phase', help='학습된 모델의 진행 단계')
    parser.add_argument('model_name', help='학습된 모델 이름')
    parser.add_argument('condition_order', help='학습조건 번호')
    parser.add_argument('model_epoch')
    parser.add_argument('--label_answer', type=bool, default=False)

    # envs parsing
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # trained args setting
    with open(f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/args_setting_{args.condition_order}.json', 'r', encoding='utf-8-sig') as f:
        trained_args_setting = json.load(f)

    # trained variable
    network = trained_args_setting['network']
    batch_size = trained_args_setting['batch_size']
    num_workers = trained_args_setting['num_workers']
    pin_memory = trained_args_setting['pin_memory']
    drop_last = trained_args_setting['drop_last']
    device_num = trained_args_setting['device_num']
    loss_name = trained_args_setting['loss_name']
    random_seed = trained_args_setting['random_seed']

    # -------------------------------------------------------------------------
    # environment setting
    utils.envs_setting(random_seed)

    # # make logger
    # log = utils.get_logger(
    #     get='TRAIN',
    #     log_file_name=f'{root_path}/log/train.log'
    # )
    # logger = utils.get_logger('train', logging_level='info')

    # class_dict
    with open(f'{args.root_path}/datasets/{args.for_test_dataset_name}/dataset_info.json', 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)

    # with open(f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/class_info.json', 'r', encoding='utf-8-sig') as f:
    #     class_info = json.load(f)
    class_dict = dataset_info['class_info']['dict']
    class_list = list(class_dict.keys())

    # color_dict
    color_dict = dataset_info['color_info']['dict']
    color_list = list(color_dict.values())

    img_path = f'{args.root_path}/datasets/{args.for_test_dataset_name}/val/images'
    if args.label_answer:
        label_path = f'{args.root_path}/datasets/{args.for_test_dataset_name}/val/label'
    else:
        label_path = img_path

    # get dataloader
    print('\n>>> get dataloader')
    Test_Dataloader = dutils.get_dataloader(
        mode='test',
        img_path=img_path,
        label_path=label_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    # -------------------------------------------------------------------------
    # device selection
    device = tutils.get_device(device_num)
    
    # model
    print('\n>>> get model...')
    result_save_path = f'{args.root_path}/{args.phase}/{args.model_name}/{args.condition_order}/{args.model_epoch}'
    model = mutils.GetModel(
        model_name=network,
        pretrained=False, 
        # n_outputs=9)
        n_outputs=len(class_list)
    )
    model.load_state_dict(torch.load(f'{result_save_path}/weight.pt'))
    model = model.to(device)

    # -------------------------------------------------------------------------
    # loss function
    if args.label_answer:
        loss_function = tutils.get_loss_function(method=loss_name)
    else:
        loss_function = None
    
    # -------------------------------------------------------------------------
    # model_test
    print('\n>>> model_test...')
    pred_label_ary, true_label_ary = tutils.model_test(
        model=model,
        test_dataloader=Test_Dataloader,
        device=device,
        loss_function=loss_function
    )
    # -------------------------------------------------------------------------
    # image save
    
    os.makedirs(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label', exist_ok=True)
    os.makedirs(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label_color', exist_ok=True)
    os.makedirs(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label_merge', exist_ok=True)
    img_name_list = Test_Dataloader.dataset.img_name_list

    # label color
    pred_label_color_ary = np.take(color_list, pred_label_ary, axis=0)

    # 라벨을 통해 label color 이미지 구하기
    row_label_list = []
    for img_name, pred_label, pred_label_color in zip(img_name_list, pred_label_ary, pred_label_color_ary):
        row = int(img_name.split('.')[0][-5:].split('_')[0])
        col = int(img_name.split('.')[0][-5:].split('_')[1])
        if col == 0:
            row_label_list.append([])
        
        row_label_list[row].append(pred_label)
        cv2.imwrite(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label_color/{img_name}', pred_label_color)
        cv2.imwrite(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label/{img_name}', pred_label)

    # merge
    col_label_list = []
    for row_label in row_label_list:
        col_label = np.concatenate(row_label, axis=1)
        col_label_list.append(col_label)
    
    pred_label_merge = np.concatenate(col_label_list, axis=0)
    pred_label_color_merge = np.take(color_list, pred_label_merge, axis=0)
    
    cv2.imwrite(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label_merge/pred_label_merge.png', pred_label_merge)
    cv2.imwrite(f'{result_save_path}/{args.for_test_dataset_name}_test_result/label_merge/pred_label_color_merge.png', pred_label_color_merge)

    # -------------------------------------------------------------------------
    if args.label_answer:
        true_label_ary = np.reshape(true_label_ary, (-1))
        pred_label_ary = np.reshape(pred_label_ary, (-1))
        test_confusion_matrix = tutils.make_confusion_matrix(
            class_list=class_list,
            true=true_label_ary,
            pred=pred_label_ary)
        utils.save_csv(save_path=f'{result_save_path}/{args.for_test_dataset_name}_test_confusion_matrix.csv',
                       data_for_save=test_confusion_matrix)

    # # result df save
    # true_class_list = []
    # pred_class_list = []
    # for true_label, pred_label, string in zip(true_label_ary, pred_label_ary, string_list):
    #     true_class = class_list[true_label]
    #     pred_class = class_list[pred_label]
    #     print(f'정답:{true_class}, 예측:{pred_class}, 문장:{string}')
    #     true_class_list.append(true_class)
    #     pred_class_list.append(pred_class)
    
    # result_df = pd.DataFrame([true_class_list, pred_class_list, Test_Dataloader.dataset.string_list], index=['true', 'pred', 'string']).T
    # utils.save_csv(save_path=f'{result_save_path}/test_result_df.csv', 
    #                data_for_save=result_df, 
    #                index=False)
    # -------------------------------------------------------------------------

    with open(f'{result_save_path}/{args.for_test_dataset_name}_test_result/dataset_info.json', 'w', encoding='utf-8-sig') as f:
        json.dump(dataset_info, f, indent='\t', ensure_ascii=False)


if __name__ == '__main__':
    main()
