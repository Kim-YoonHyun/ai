# common module
import numpy as np
import os
import sys
import argparse
import time
import copy
from tqdm import tqdm

# model
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

# data
from sklearn.utils import shuffle

sys.path.append('/home/kimyh/python/ai')
from utilsmodule import utils
from mylocalmodules import data_loader as dutils
from mylocalmodules import model as mutils
from mylocalmodules import train as tutils

# my local module
# from mylocalmodule import utils as lutils
# from mylocalmodule import efficientnetutils as eutils
# my global module
# from myglobalmodule import utils as gutils


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
    
    # 데이터 + label 리스트 초기화
    train_data_with_label = []
    test_data_with_label = []

    # 경로에서 데이터 불러오기
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
    
    # 데이터 셔플
    train_data_with_label = np.array(train_data_with_label)
    train_data_with_label_shuffled = shuffle(train_data_with_label, random_state=random_seed)

    # 라벨데이터를 제외한 데이터만 인덱싱
    train_data = train_data_with_label_shuffled[:, :, :, :-1]

    # 학습데이터 비율 값 계산
    train_num = int(len(train_data)*0.8)

    # 학습 라벨 인덱싱
    train_label = []
    for i in train_data_with_label_shuffled[:, :, :, -1:]:
        train_label.append(int(i[0][0][0]))
    train_label = np.array(train_label)

    # val data split
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


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def get_output(model, loss_function, data, label, device, sanity_check=False, optimizer=None):
    """
    모델에 데이터를 입력하여 얻어낸 결과를 통해 loss 를 구하고
    model weight 를 학습시키는 코드.
    optimizer 유무를 통해 validation 에 활용 가능

    parameters
    ----------
    model: torch model
        학습에 활용할 네트워크 모델

    loss_function: torch.nn.modules.loss.<function>
        학습에 활용할 loss function

    data: numpy array
        batch 화 된 학습 데이터. (batch_lenght, batch_size, ct_image_page_number, img_size, img_size)

    label: numpy array
        batch 화 된 학습 라벨. (batch_length, batch_size)

    device: cuda or cpu
        학습을 진행할 장치

    optimizer: torch.optim.<optimizer>
        학습에 활용할 optimizer

    returns
    -------
    loss: float
        데이터 입력후 계산되어지는 loss 값    
    """
    # loss 변수 초기화
    running_loss = 0.0
    len_data = 0
    
    # 배치별로 결과 계산
    for xb, yb in zip(data, label):
        xb = torch.Tensor(xb)
        yb = torch.Tensor(yb)
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb.type(torch.int64)
        
        output = model(xb)
        loss_b = loss_function(output, yb)
        
        if optimizer:
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
        
        loss_b = loss_b.item()
        running_loss += loss_b
        len_data += len(xb)

        if sanity_check is True:
            break

    # loss 계산
    loss = running_loss/len_data
    return loss


def train(model, params, device):
    '''
    efficient net 을 통해 3d image 를 학습시키는 코드

    parameters
    ----------
    model: torch model
        학습을 진행할 네트워크 모델

    params: dictionary
        학습에 활용할 변수를 모은 dictionary
        ----------
        epochs: int
            학습 에포크 값

        loss_function: torch.nn.modules.loss.<function>
            학습에 활용할 loss function

        optimizer: torch.optim.<optimizer>
            학습에 활용할 optimizer

        train_data: numpy array
            학습용 데이터. (data_n, ct_image_page_number, img_size, img_size)

        train_label: numpy array
            학습 데이터 라벨. (data_n, )

        val_data: numpy array
            학습시 validation 용 데이터. (data_n, ct_image_page_number, img_size, img_size)

        val_label: numpy array
            학습 validation 라벨. (data_n, )

        batch_size: int
            학습 batch

        lr_scheduler: torch.optim.lr_scheduler.<scheduler>
            learning rate 를 조정할 scheduler
        
        model_save_path: str
            모델 및 결과를 저장할 폴더 경로(자동생성)
    
    device: cuda or cpu
        학습을 진행할 장치

    returns
    -------
    model: pytorch model
        최적의 학습 weigth가 적용된 학습 모델
    
    history: json
        각 epoch 별 학습 결과 및 최적의 결과값
    '''

    # 변수 지정
    epochs = params['epochs']
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    train_data = params['train_data']
    train_label = params['train_label']
    val_data = params['val_data']
    val_label = params['val_label']
    batch_size = params['batch_size']
    lr_scheduler = params['lr_scheduler']
    model_save_path = params['model_save_path']

    # 초기 bset loss 값 설정(무한대 값)
    best_loss = float('inf')

    # history dict 초기화
    history = {'best':{'epoch':0, 'loss':0}}
    start_time = time.time()
    
    # batch 분배
    train_data_b = lutils.get_batch(batch_size, train_data)
    train_label_b = lutils.get_batch(batch_size, train_label)
    val_data_b = lutils.get_batch(batch_size, val_data)
    val_label_b = lutils.get_batch(batch_size, val_label)

    # epoch 진행
    for epoch in range(epochs):
        current_lr = get_lr(optimizer)
        print(f'Epoch {epoch + 1}/{epochs}, current lr={current_lr:.4f}')
        history[f'epoch {epoch+1}'] = {'train_loss':0, 'val_loss':0}

        # 학습
        model.train()
        train_loss = get_output(model, loss_function, train_data_b, train_label_b, device, optimizer)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = get_output(model, loss_function, val_data_b, val_label_b, device)

        # 최적의 epoch, loss, weight 저장
        if val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            history['best']['epoch'] = best_epoch
            history['best']['loss'] = best_loss
        
        # learning rate 감소(조건부)
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        # 결과 표시 및 이력 저장
        print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, time: {(time.time()-start_time)/60:.4f} min')
        print('-'*10)
        history[f'epoch {epoch+1}']['train_loss'] = train_loss
        history[f'epoch {epoch+1}']['val_loss'] = train_loss

    # 최적의 모델 저장
    print(f'best: {best_epoch}')
    model.load_state_dict(best_model_wts)
    gutils.createfolder(model_save_path)
    torch.save(model.state_dict(), f'{model_save_path}/batch{batch_size}_epoch{str(best_epoch).zfill(4)}.pt')

    return model, history


def main():
    # parsing
    parser = argparse.ArgumentParser()

    # path parsing
    parser.add_argument('root_path', help='project directory path') 
    parser.add_argument('phase', help='project phase')
    parser.add_argument('dataset_name', help='dataset for training')
    parser.add_argument('--network', default='efficientnet', help='training network')

    # data loader parsing
    parser.add_argument('--batch_size', type=int, default=8, help='data batch size')

    # train parsing
    parser.add_argument('--device_num', type=int, help='사용할 device 번호')
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--width_coef', type=float, default=1.5, help='model width')
    parser.add_argument('--depth_coef', type=float, default=1.5, help='model depth')
    parser.add_argument('--scale', type=float, default=1.0, help='image scale')

    

    parser.add_argument('--initial_learning_rate', type=float, default=0.01, help='training learning rate')
    parser.add_argument('--lr_descending_rate', type=float, default=0.7, help='learning rate descending rate')
    parser.add_argument('--lr_descending_step', type=int, default=10, help='learning rate descending step')
    parser.add_argument('--normalize', type=bool, default=False, help='data normalize or not')

    # envs parsing
    parser.add_argument('--random_seed', type=int, default=42, help='random seed / default=42')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # envs setting
    utils.envs_setting(args.random_seed)

    # envs setting 
    utils.envs_setting(args.random_seed)

    # logger
    logger = utils.get_logger('train', logging_level='info')

    # -------------------------------------------------------------------------
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
        norm_train_data = lutils.normalize_3D(train_data)
        norm_val_data = lutils.normalize_3D(val_data)
        norm_test_data = lutils.normalize_3D(test_data)
        model_save_path = f'{args.root_path}/{args.phase}/norm_{args.dataset_name}_model/{args.network}_w{args.width_coef}_d{args.width_coef}_s{args.scale}'
    else:
        model_save_path = f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}_w{args.width_coef}_d{args.width_coef}_s{args.scale}'

    # cuda gpu select
    device = gutils.get_device(1)

    # model
    model = eutils.efficientnet(initial_channel=96, 
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
    test_data_b = lutils.get_batch(args.batch_size, norm_test_data)
    test_label_b = lutils.get_batch(args.batch_size, test_label)
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
        
    # save acc df
    confusion_matrix = gutils.make_confusion_matrix(class_list=class_list, 
                         true=test_label, 
                         pred=test_pred_label, 
                         save=f'{model_save_path}/{best_model_name}_test_result.csv')

    print('\n')
    print(confusion_matrix)


# check
if __name__ == '__main__':
    main()
    # sys.exit()

    # # class dictionary
    # class_dict = {'AML':0, 'else':1}
    # num_classes = len(class_dict)
    # class_list = list(class_dict.keys())

    
    # root_path = '/home/kimyh/ai/image/classification/kidney_cancer'
    # dataset_name = '02_PRE_3D_AML_else'
    # dataset_path=f'{root_path}/datasets/{dataset_name}'

    # train_data = []
    # train_label = []
    # for tp in ['train_data', 'test_data']:
    #         class_list = os.listdir(f'{dataset_path}/{tp}')
    #         for class_name in class_list:
    #             img_list = os.listdir(f'{dataset_path}/{tp}/{class_name}')
    #             for img_name in tqdm(img_list[:10]):
                    
    #                 img = np.load(f'{dataset_path}/{tp}/{class_name}/{img_name}')
    #                 # img = preprocess(img)
    #                 label = class_dict[class_name]
                    
    #                 # # class_dummy_ary = np.full((img.shape[0], img.shape[1], 1), class_dict[class_name])
    #                 # img = np.concatenate((img, class_dummy_ary), axis=-1)
    #                 if tp == 'train_data':
    #                     train_data.append(img)
    #                     train_label.append(label)

    #                     # train_data_with_label.append(img)
    #                 # if tp == 'test_data':
    #                 #     test_data_with_label.append(img)
    # train_data = np.array(train_data)
    # train_label = np.array(train_label)
        

    # train_point_cloud = []
    # for data in tqdm(train_data):
    #     point_data = []
    #     position = np.where(data > 0)
    #     x_position = np.expand_dims(position[0], axis=-1)
    #     y_position = np.expand_dims(position[1], axis=-1)
    #     z_position = np.expand_dims(position[2], axis=-1)
    #     color = np.expand_dims(data[np.where(data > 0)], axis=-1)
    #     temp = np.concatenate((x_position, y_position, z_position, color), axis=1)
    #     train_point_cloud.append(temp)
    # train_point_cloud = np.array(train_point_cloud)
    # for i in train_point_cloud:
    #     print(i.shape)
        