import numpy as np
import copy
import time
import sys

import torch

sys.path.append('/home/kimyh/ai')
from myglobalmodule.utils import createfolder

def normalize_3D(data):
    '''
    입력한 3D data를 image 한 장 단위로 normalize 하는 함수

    parameters
    ----------
    data: numpy array
        3D image 데이터 집합체. (data_n, image_n, image_size, image_size)
    
    returns
    -------
    norm_data: numpy array
        same as input data. normalize 된 image 데이터 집합체
    
    '''
    from tqdm import tqdm
    norm_data = []
    for idx, img_3d in enumerate(tqdm(data)):
        norm_data.append([])
        for img in img_3d:
            if np.max(img) != 0.0:
                aver = np.average(img)
                std = np.std(img)
                img = np.divide(np.subtract(img, aver), std)
            norm_data[idx].append(img)
    norm_data = np.array(norm_data)
    return norm_data






def get_batch(batch_size, data):
    '''
    학습데이터를 batch 화 시키는 함수

    parameters
    ----------
    batch_size: int
        데이터에 적용할 batch size

    data: numpy array
        batch 를 적용할 데이터. (data_n, image_n, image_size, image_size)

    returns
    -------
    data_b: list
        batch 가 적용된 데이터.(batch_lenght, batch_size, image_n, image_size, image_size)
        batch_lenght X batch_size = data_n.
    '''

    batch_len = len(data) // batch_size

    data_b = []
    for idx in range(batch_len + 1):
        batched = data[batch_size*idx:batch_size*(idx+1)]
        if len(batched) != 0:
            data_b.append(batched)

    return data_b





