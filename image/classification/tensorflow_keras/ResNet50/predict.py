import sys
import os
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

sys.path.append('/home/kimyh/ai')
from myglobalmodule.utils import make_acc_df
from myglobalmodule.utils import make_acc_table
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# 전역변수
ROOT_PATH = '/home/kimyh/ai/image/classification/kidney_cancer'
MODEL = '6th/pre2_model/ResNet50/img64_batch16_epochs100_model'
DATASET_NAME = 'pre2'
IMG_SIZE = 64
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 학습된 모델 불러오기
    trained_model = tf.keras.models.load_model(f'{ROOT_PATH}/{MODEL}')    
    
    # 클래스 불러오기
    class_list = os.listdir(f'{ROOT_PATH}/datasets/{DATASET_NAME}/pred_data')
    num_classes = len(class_list)
    
    # 평가
    print('predict...')
    true_label_list = []
    output_label_list = []
    
    for idx, class_name in enumerate(class_list):
        print(class_name)
        image_list = os.listdir(f'{ROOT_PATH}/datasets/{DATASET_NAME}/pred_data/{class_name}')
        for image_name in image_list:
            true_label_list.append(idx)
            img = cv2.imread(f'{ROOT_PATH}/datasets/{DATASET_NAME}/pred_data/{class_name}/{image_name}')
            img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE))
            img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
            img = img / 255
            output = trained_model.predict(img)
            output_label_list.append(output.argmax())
    
    try:
        os.makedirs(f'{ROOT_PATH}/{MODEL}_predict')
    except:
        pass

    acc_table = make_acc_table(class_list=class_list,
                               true_label_list=true_label_list,
                               output_label_list=output_label_list,
                               save_path=f'{ROOT_PATH}/{MODEL}_predict/{DATASET_NAME}_predict_report.json')
    
    acc_df = make_acc_df(acc_table=acc_table, class_list=class_list,
                         save_path=f'{ROOT_PATH}/{MODEL}_predict/{DATASET_NAME}_predict_report.csv')