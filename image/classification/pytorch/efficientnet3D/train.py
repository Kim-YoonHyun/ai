# common module
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

# model
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

# data
from sklearn.utils import shuffle

# my local module
from mylocalmodule.utils import train
from mylocalmodule.utils import get_batch
from mylocalmodule.utils import normalize_3D
from mylocalmodule.efficientnetutils import efficientnet

# my global module
sys.path.append('/home/kimyh/ai')
from myglobalmodule.utils import save_json
from myglobalmodule.utils import make_acc_df
from myglobalmodule.utils import envs_setting
from myglobalmodule.utils import get_device


def make_dataset(dataset_path, class_dict, random_seed):
    '''
    dataset_path 를 입력받아 내부의 구조에 따라 데이터를 불러들여
    각각 train, val, test 로 분할하는 함수
    

    parameters
    ----------
    dataset_path: str
        데이터셋의 경로 (데이터셋 이름 포함)

    class_dict: dict
        데이터의 클래스 이름과 클래스 label 이 정의된 dictionary.
        {class_name1 : class_label1, ...}

    random_seed: int
        데이터 분할 시 적용할 random seed
    ----------


    returns
    -------
    *_data: numpy array
        분할된 이미지 데이터. (data_n, ct_image_page_number, img_size, img_size)

    *_label: numpy array
        각 데이터의 라벨 값. (data_n, )
    -------

    '''
    import os
    from tqdm import tqdm
    import numpy as np
    
    train_data_with_label = []
    test_data_with_label = []
    for tp in ['train_data', 'test_data']:
        class_list = os.listdir(f'{dataset_path}/{tp}')
        for class_name in class_list:
            img_list = os.listdir(f'{dataset_path}/{tp}/{class_name}')
            for img_name in tqdm(img_list):
                
                img = np.load(f'{dataset_path}/{tp}/{class_name}/{img_name}')
                class_dummy_ary = np.full((img.shape[0], img.shape[1], 1), class_dict[class_name])
                img = np.concatenate((img, class_dummy_ary), axis=-1)
                if tp == 'train_data':
                    train_data_with_label.append(img)
                if tp == 'test_data':
                    test_data_with_label.append(img)
    
    # train & val data
    train_data_with_label = np.array(train_data_with_label)
    train_data_with_label_shuffled = shuffle(train_data_with_label, random_state=random_seed)
    train_data = train_data_with_label_shuffled[:, :, :, :-1]
    train_num = int(len(train_data)*0.8)

    train_label = []
    for i in train_data_with_label_shuffled[:, :, :, -1:]:
        train_label.append(int(i[0][0][0]))
    train_label = np.array(train_label)

    val_data = train_data[train_num:, :, :, :]
    val_label = train_label[train_num:]
    train_data = train_data[:train_num, :, :, :]
    train_label = train_label[:train_num]

    # test data
    test_data_with_label = np.array(test_data_with_label)
    test_data_with_label_shuffled = shuffle(test_data_with_label, random_state=random_seed)
    test_data = test_data_with_label_shuffled[:, :, :, :-1]

    test_label = []
    for i in test_data_with_label_shuffled[:, :, :, -1:]:
        test_label.append(int(i[0][0][0]))
    test_label = np.array(test_label)

    return train_data, train_label, val_data, val_label, test_data, test_label


def main():
    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='project directory path') 
    parser.add_argument('phase', help='project phase')   
    parser.add_argument('dataset_name', help='dataset for training')
    parser.add_argument('--network', default='efficientnet', help='training network')
    parser.add_argument('--width_coef', type=float, default=1.5, help='model width')
    parser.add_argument('--depth_coef', type=float, default=1.7, help='model depth')
    parser.add_argument('--scale', type=float, default=1.5, help='image scale')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed / default=42')
    parser.add_argument('--epochs', type=int, default=150, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='data batch size')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01, help='training learning rate')
    parser.add_argument('--lr_descending_rate', type=float, default=0.7, help='learning rate descending rate')
    parser.add_argument('--lr_descending_step', type=int, default=10, help='learning rate descending step')
    parser.add_argument('--normalize', type=bool, default=True, help='data normalize or not')
    args = parser.parse_args()

    # environmet setting
    envs_setting(args.random_seed)

    # class dictionary
    class_dict = {'AML':0, 'else':1}
    num_classes = len(class_dict)
    class_list = list(class_dict.keys())

    # separate dataset
    print('\n>>> make datasets...')
    train_data, train_label, val_data, val_label, test_data, test_label = make_dataset(
        dataset_path=f'{args.root_path}/datasets/{args.dataset_name}',
        class_dict=class_dict,
        random_seed=args.random_seed)

    # normalize
    if args.normalize:
        print('\n>>> dataset_normalize...')
        norm_train_data = normalize_3D(train_data)
        norm_val_data = normalize_3D(val_data)
        norm_test_data = normalize_3D(test_data)
        model_save_path = f'{args.root_path}/{args.phase}/norm_{args.dataset_name}_model/{args.network}_w{args.width_coef}_d{args.width_coef}_s{args.scale}'
    else:
        model_save_path = f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}_w{args.width_coef}_d{args.width_coef}_s{args.scale}'

    # cuda gpu select
    device = get_device(1)

    # model
    model = efficientnet(initial_channel=96, 
                         num_classes=num_classes, 
                         width_coef=args.width_coef, 
                         depth_coef=args.depth_coef, 
                         scale=args.scale
                         ).to(device)
    summary(model, (96, 128, 128), device=device.type)

    # loss function, optimizer, lr_scheduler
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.initial_learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min', 
                                     factor=args.lr_descending_rate, 
                                     patience=args.lr_descending_step)
    
    # define the training parameters
    train_variable = {
        'epochs':args.epochs,
        'optimizer':optimizer,
        'loss_function':loss_function,
        'train_data':norm_train_data,
        'train_label':train_label,
        'val_data':norm_val_data,
        'val_label':val_label,
        'batch_size':args.batch_size,
        'class_list':class_list,
        'lr_scheduler':lr_scheduler,
        'model_save_path':model_save_path
    }
    
    print('\n>>> train...')
    trained_model, train_history = train(model, train_variable, device)

    print('\n>>> test...')
    test_data_b = get_batch(args.batch_size, norm_test_data)
    test_label_b = get_batch(args.batch_size, test_label)
    trained_model.eval()
    with torch.no_grad():
        test_pred_label = []
        for xb, yb in zip(test_data_b, test_label_b):
            xb = torch.Tensor(xb)
            yb = torch.Tensor(yb)
            xb = xb.to(device)
            yb = yb.to(device)
            
            test_pred_b = trained_model(xb)
            test_pred_b = test_pred_b.to('cpu').detach().numpy()
            
            for test_pred in test_pred_b:
                test_pred_label.append(np.argmax(test_pred))

    # save train history
    best_model_name = f'batch{args.batch_size}_epoch{str(train_history["best"]["epoch"]).zfill(4)}'
    save_json(save_path=f'{model_save_path}/{best_model_name}_train_history.json', data_for_save=train_history)
    
    # save acc df
    acc_df = make_acc_df(class_list=class_list, 
                         true=test_label, 
                         pred=test_pred_label, 
                         save=f'{model_save_path}/{best_model_name}_test_result.csv')

    print('\n')
    print(acc_df)


# check
if __name__ == '__main__':
    main()
    sys.exit()

    # class dictionary
    class_dict = {'AML':0, 'else':1}
    num_classes = len(class_dict)
    class_list = list(class_dict.keys())

    
    root_path = '/home/kimyh/ai/image/classification/kidney_cancer'
    dataset_name = '02_PRE_3D_AML_else'
    dataset_path=f'{root_path}/datasets/{dataset_name}'

    train_data = []
    train_label = []
    for tp in ['train_data', 'test_data']:
            class_list = os.listdir(f'{dataset_path}/{tp}')
            for class_name in class_list:
                img_list = os.listdir(f'{dataset_path}/{tp}/{class_name}')
                for img_name in tqdm(img_list[:10]):
                    
                    img = np.load(f'{dataset_path}/{tp}/{class_name}/{img_name}')
                    # img = preprocess(img)
                    label = class_dict[class_name]
                    
                    # # class_dummy_ary = np.full((img.shape[0], img.shape[1], 1), class_dict[class_name])
                    # img = np.concatenate((img, class_dummy_ary), axis=-1)
                    if tp == 'train_data':
                        train_data.append(img)
                        train_label.append(label)

                        # train_data_with_label.append(img)
                    # if tp == 'test_data':
                    #     test_data_with_label.append(img)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
        

    train_point_cloud = []
    for data in tqdm(train_data):
        point_data = []
        position = np.where(data > 0)
        x_position = np.expand_dims(position[0], axis=-1)
        y_position = np.expand_dims(position[1], axis=-1)
        z_position = np.expand_dims(position[2], axis=-1)
        color = np.expand_dims(data[np.where(data > 0)], axis=-1)
        temp = np.concatenate((x_position, y_position, z_position, color), axis=1)
        train_point_cloud.append(temp)
    train_point_cloud = np.array(train_point_cloud)
    for i in train_point_cloud:
        print(i.shape)