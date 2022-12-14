import sys
import os
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# 전역변수
class_list = ['AML', 'non_AML']
IMG_SIZE = 64
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 학습된 모델 불러오기
    trained_model = tf.keras.models.load_model('./trained_model')    
    
    # 평가

    result_dict = {}
    
    print('predict...')
    folder_list = os.listdir('./image_sample')
    for folder_name in folder_list:
        AML_count = 0
        else_count = 0
        img_list = os.listdir(f'./image_sample/{folder_name}')
        for img_name in img_list:
            img = cv2.imread(f'./image_sample/{folder_name}/{img_name}')
            # img = img.astype(np.float32)
            img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE))
            img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
            img = img / 255
            output = trained_model.predict(img)
            # print(img_name, class_list[output.argmax()])
            
            if output.argmax() == 0:
                AML_count += 1
            else:
                else_count += 1

        if AML_count > else_count:
            print(f'{folder_name}, [AML{AML_count:2d}, non AML{else_count:2d}]  ---> AML')
        elif AML_count == else_count:
            print(f'{folder_name}, [AML{AML_count:2d}, non AML{else_count:2d}]  ---> 50:50')
        else:
            print(f'{folder_name}, [AML{AML_count:2d}, non AML{else_count:2d}]  ---> non AML')
        
        result_dict[folder_name[:10]] = {'AML':AML_count,
                                        'else':else_count}
        

        
        
        
        
        
        
        