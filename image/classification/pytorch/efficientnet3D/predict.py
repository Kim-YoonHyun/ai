# common module
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

# model
import torch

# my local module
from mylocalmodule.utils import get_batch
from mylocalmodule.utils import normalize_3D
from mylocalmodule.efficientnetutils import efficientnet

# my global module
sys.path.append('/home/kimyh/ai')
from myglobalmodule.utils import make_acc_df
from myglobalmodule.utils import envs_setting
from myglobalmodule.utils import get_device
from myglobalmodule.utils import see_device
from myglobalmodule.utils import createfolder


def make_dataset_for_predict(dataset_path, class_dict):
    '''
    dataset_path 를 입력받아 내부의 구조에 따라 데이터를 불러들여
    test_data 화 시키는 함수

    parameters
    ---------------------------------------------------------
    dataset_path
    - type: str
    - description: 데이터셋의 경로 (데이터셋 이름 포함)

    class_dict
    - type: dict
    - shape: {class_name1 : class_label1, ...}
    - description: 데이터의 클래스 이름과 클래스 label 이 정의된 dictionary
    ---------------------------------------------------------


    return
    ---------------------------------------------------------
    test_data
    - type: numpy array
    - shape: (data_n, ct_image_page_number, img_size, img_size)
    - description: 학습용으로 정제된 데이터

    test_label
    - type: numpy array
    - shape: (data_n, )
    - description: 각 데이터의 라벨 값
    ---------------------------------------------------------

    '''
    from tqdm import tqdm
    import numpy as np
    
    test_data_with_label = []
    for tp in ['train_data', 'test_data']:
        class_list = os.listdir(f'{dataset_path}/{tp}')
        for class_name in class_list:
            img_list = os.listdir(f'{dataset_path}/{tp}/{class_name}')
            for img_name in tqdm(img_list):
                
                img = np.load(f'{dataset_path}/{tp}/{class_name}/{img_name}')
                class_dummy_ary = np.full((img.shape[0], img.shape[1], 1), class_dict[class_name])
                img = np.concatenate((img, class_dummy_ary), axis=-1)
                test_data_with_label.append(img)
    
    # test data
    test_data_with_label = np.array(test_data_with_label)
    test_data = test_data_with_label[:, :, :, :-1]

    test_label = []
    for i in test_data_with_label[:, :, :, -1:]:
        test_label.append(int(i[0][0][0]))
    test_label = np.array(test_label)

    return test_data, test_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='project directory path') 
    parser.add_argument('trained_model', help='trained model for predict')
    parser.add_argument('dataset_name', help='dataset for predict')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--normalize', type=bool, default=True)
    args = parser.parse_args()

    # envs_setting
    envs_setting(args.random_seed)

    # class
    class_dict = {'AML':0, 'else':1}
    num_classes = len(class_dict)
    class_list = list(class_dict.keys())

    # dataset
    print('\n>>> make datasets...')
    test_data, test_label = make_dataset_for_predict(
        dataset_path=f'{args.root_path}/datasets/{args.dataset_name}',
        class_dict=class_dict)

    # normalize
    if args.normalize:
        print('\n>>> dataset_normalize...')
        
        norm_test_data = normalize_3D(test_data)
        model_save_path = f'{args.root_path}/{args.trained_model[:-3]}/norm_{args.dataset_name}_predict'
    else:
        model_save_path = f'{args.root_path}/{args.trained_model[:-3]}/{args.dataset_name}_predict'
    
    device = get_device(1)
    width_coef = float(args.trained_model.split('/')[2].split('_')[1][1:])
    depth_coef = float(args.trained_model.split('/')[2].split('_')[2][1:])
    scale = float(args.trained_model.split('/')[2].split('_')[3][1:])
    model = efficientnet(initial_channel=96, 
                        num_classes=num_classes, 
                        width_coef=width_coef, 
                        depth_coef=depth_coef, 
                        scale=scale
                        ).to(device)
    model.load_state_dict(torch.load(f'{args.root_path}/{args.trained_model}'))
    trained_model = model

    print('\n>>> predict...')
    test_data_b = get_batch(args.batch_size, norm_test_data)
    test_label_b = get_batch(args.batch_size, test_label)
    trained_model.eval()
    with torch.no_grad():
        test_pred_label = []
        idx = 0
        for xb, yb in zip(test_data_b, test_label_b):
            xb = torch.Tensor(xb)
            yb = torch.Tensor(yb)
            xb = xb.to(device)
            yb = yb.to(device)
            
            test_pred_b = trained_model(xb)
            test_pred_b = test_pred_b.to('cpu').detach().numpy()
            
            for test_pred in test_pred_b:
                test_pred_label.append(np.argmax(test_pred))
            idx += 1
    
    # save acc df
    print('\n>>> save predict result...')
    createfolder(model_save_path)
    acc_df = make_acc_df(class_list=class_list, 
                         true=test_label, 
                         pred=test_pred_label, 
                         save=f'{model_save_path}.csv')
    print(acc_df)
    

# check
if __name__ == '__main__':
    main()