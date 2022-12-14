import os
import sys
import json
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import Xception

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import ResNet50V2
# from tensorflow.keras.applications import MobileNetV3Small
# from tensorflow.keras.applications import MobileNetV3Large
# from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Large
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
                                   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# 환경 세팅
def setting():
    import tensorflow as tf
    print(f'tensorflow version: {tf.__version__}')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # Num GPUs Available:  1
    print(tf.config.list_physical_devices('GPU'))
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'



def make_generator(args, img_data_path, dataset_type, mode):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if mode == 'train':
        Img_Gen = ImageDataGenerator(
            # rotation_range=5,
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # brightness_range=[.9, .9],
            horizontal_flip=True,   # input을 무작위로 수평으로 뒤집음
            rescale=1. / 255,       # 데이터에 곱할 값
            validation_split=.1     # validation 비율 
            )
        Train_Gen = Img_Gen.flow_from_directory(
            f'{img_data_path}/{dataset_type}',
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            subset='training'
            )
        return Train_Gen

    if mode == 'validation':
        Img_Gen = ImageDataGenerator(
            rescale=1. / 255,
            )
        Validation_Gen = Img_Gen.flow_from_directory(
            f'{img_data_path}/{dataset_type}',
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            #     subset='validation'    
            )
        return Validation_Gen
    


def get_model(args):

    num_classes = len(os.listdir(f'{args.root_path}/datasets/{args.dataset_name}/train_data'))
    
    # 학습 모델 구축
    model = Sequential()
    if args.network == 'ResNet50':
        model.add(ResNet50(include_top=True, weights=None, input_shape=(args.img_size, args.img_size, 3), classes=num_classes))
    if args.network == 'Xception':
        model.add(Xception(include_top=True, weights=None, input_shape=(args.img_size, args.img_size, 3), classes=num_classes))
    if args.network == 'VGG16':
        model.add(VGG16(include_top=True, weights=None, input_shape=(args.img_size, args.img_size, 3), classes=num_classes))
    # if args.network == 'MobileNetV3Large':
    #     model.add(MobileNetV3Large(include_top=True, weights=None, input_shape=(args.img_size, args.img_size, 3), classes=class_num))
    # if args.network == 'NASNetMobile':
    #     model.add(NASNetMobile(include_top=True, weights=None, input_shape=(args.img_size, args.img_size, 3), classes=class_num))
    
    # 학습 모델 요약
    model.summary()

    # 학습 모델 구성
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



def train(args, model, Train_Gen, Validation_Gen):
    # 학습
    History = model.fit_generator(
        Train_Gen,
        epochs=args.epochs,
        # steps_per_epoch= 10,
        steps_per_epoch=Train_Gen.samples / args.batch_size,
        validation_data=Validation_Gen,
        validation_steps=Validation_Gen.samples / args.batch_size,
        # steps_per_epoch= 128
        )
    
    # 학습된 모델 저장
    model.save(f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}/img{args.img_size}_batch{args.batch_size}_epochs{args.epochs}_model')
    
    # 학습 결과    
    history_dict = History.history
    
    # 저장용 폴더 생성
    try:
        os.makedirs(f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}')
    except:
        pass
    
    # 학습 내역 json 저장을 위해 모든 값을 float32 에서 float으로 변경
    for key, value in history_dict.items():
        history_dict[key] = list(map(float, value))
    
    # 학습 history 저장
    with open(f'{args.root_path}/{args.phase}/{args.dataset_name}_model/{args.network}/img{args.img_size}_batch{args.batch_size}_epochs{args.epochs}_model_history.json', 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent='\t', ensure_ascii=False)



def main():
    parser = argparse.ArgumentParser(description='이미지 학습')
    parser.add_argument('root_path', help='학습을 진행할 프로젝트 폴더 경로')
    parser.add_argument('phase', help='학습 단계')
    parser.add_argument('dataset_name', help='학습을 진행할 데이터셋 이름')
    parser.add_argument('--network', default='ResNet50', help='학습에 활용할 네트워크 / default=ResNet50')
    parser.add_argument('--img_size', type=int, default=64, help='학습시 재조정할 이미지 사이즈 / default=64')
    parser.add_argument('--batch_size', type=int, default=16, help='학습 배치 사이즈 / default=16')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 / default=100')
    args = parser.parse_args()
    
    setting()
    
    Train_Gen = make_generator(
        args=args,
        img_data_path=f'{args.root_path}/datasets/{args.dataset_name}', 
        dataset_type='train_data',
        mode='train')
    Validation_Gen = make_generator(
        args=args,
        img_data_path=f'{args.root_path}/datasets/{args.dataset_name}', 
        dataset_type='train_data',
        mode='validation')
    
    model = get_model(args)
    
    start = time.time()
    train(args=args,
          model=model,
          Train_Gen=Train_Gen,
          Validation_Gen=Validation_Gen)
    end = time.time()
    print(f'학습시간: {end - start} 초')
    
    
if __name__=='__main__':
    main()