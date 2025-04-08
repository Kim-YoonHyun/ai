'''
pip install pandas
pip install -U scikit-learn
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


def get_max_2nd_n_reliability(pred):
    pred_min = np.expand_dims(np.min(pred, axis=1), axis=1)
    pred = pred - pred_min
    pred_max = np.expand_dims(np.max(pred, axis=1), axis=1)
    pred = pred/pred_max

    # 1순위 예측값 없애기
    pred = np.where(pred == 1, -100, pred)

    # 2순위 예측
    max_2nd_index = np.argmax(pred, axis=1)

    # 신뢰도 구하기
    pred_reliability = (1 - np.max(pred, axis=1))*100
    return max_2nd_index, pred_reliability


def make_acc_report(uni_class_list, true, pred, reset_class=False):
    '''
    sklearn 의 classification_report 의 결과에 confusion matrix 를 더한 
    json 형태의 결과 데이터를 얻어내는 함수.

    parameters
    ----------
    uni_class_list: str list
        결과를 계산할 class list

    true: int list
        예측 결과 계산에 활용할 true 라벨 데이터

    pred: int list
        예측 결과 계산에 활용할 predict 라벨 데이터

    returns
    -------
    acc_report: json
        각 class 별 정확도 및 정확도 matrix 가 포함된 json 형태의 결과값
    '''
    try:
        true_label_list = true
        pred_label_list = pred
    except TypeError:
        encoder = LabelEncoder()
        true_label_list = encoder.fit_transform(true) 
        pred_label_list = encoder.fit_transform(pred) 

    # matrix
    matrix = np.zeros([len(uni_class_list), len(uni_class_list)])
    for t, o in zip(true_label_list, pred_label_list):
        matrix[t][o] += 1
    
    if reset_class:
        # 각 결과의 유니크 값만 정리
        unique_true_label_list = np.unique(true_label_list)
        unique_pred_label_list = np.unique(pred_label_list)

        # 합집합
        unique_label_list = list(set(unique_true_label_list).union(set(unique_pred_label_list)))

        # 존재하는 라벨값으로만 재구성
        uni_class_ary = np.array(uni_class_list)
        uni_class_list = uni_class_ary[unique_label_list].tolist()

        # 존재하는 라벨값으로만 matrix 재구성    
        matrix = np.array(matrix)
        matrix = matrix[unique_label_list].T
        matrix = matrix[unique_label_list].T

    # 결과 json 생성
    acc_report = classification_report(
        true_label_list, 
        pred_label_list, 
        output_dict=True, 
        target_names=uni_class_list
    )
    acc_report['matrix'] = matrix.tolist()

    return acc_report, uni_class_list


def make_confusion_matrix(uni_class_list, true, pred, reset_class=False):
    '''
    make_acc_json 함수의 결과 데이터로 pandas DataFrame 기반의 result table 을 만드는 함수.
    경로 설정시 .csv 형태로 저장

    parameters
    ----------
    uni_class_list: str list
        결과를 계산할 class list

    true: int list
        예측 결과 계산에 활용할 true 라벨 데이터

    pred: int list
        예측 결과 계산에 활용할 predict 라벨 데이터

    save: str
        result table 을 저장할 경로 및 이름. default=None (결과저장 X)

    returns
    -------
    confusion_matrix: pandas dataframe, csv
        학습 결과를 가독성이 좋은 형태로 변경한 dataframe. 결과 저장시 csv 로 저장됨.
    '''
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    acc_report, uni_class_list = make_acc_report(
        uni_class_list=uni_class_list, 
        true=true,
        pred=pred,
        reset_class=reset_class
    )
    for e, accs in acc_report.items():
        if e == 'accuracy':
            accuracy_list[0] = accs
            break
        accuracy_list.append(None)
        precision_list.append(accs['precision'])
        recall_list.append(accs['recall'])
        f1_list.append(accs['f1-score'])
        support_list.append(accs['support'])

    matrix = acc_report['matrix']
    df1 = pd.DataFrame(matrix, index=uni_class_list, columns=uni_class_list)
    df2 = pd.DataFrame([accuracy_list, precision_list, recall_list, f1_list, support_list], columns=uni_class_list, index=['accuracy', 'precision', 'recall', 'f1', 'support']).T
    confusion_matrix = pd.concat([df1, df2], axis=1)

    return confusion_matrix