'''
pip install pandas
'''
import sys
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


def matrix2confusion(matrix, uni_label_list, round_num=4, show_percentage=True):
    whole_sum = np.sum(matrix)
    true_sum_list = np.sum(matrix, axis=-1).tolist()
    pred_sum_list = np.sum(matrix, axis=-2).tolist()
    
    # make matrix
    if show_percentage:
        per_num = 100
    else:
        per_num = 1
    correct_sum = 0
    for i in range(len(matrix)):
        correct_count = matrix[i][i]
        correct_sum += correct_count
        pred_sum = pred_sum_list[i]
        true_sum = true_sum_list[i]
        
        # precision
        try:
            precision = correct_count / pred_sum
            precision = np.round(precision, round_num) * per_num
        except ZeroDivisionError:
            precision = None
            
        # recall
        try:
            recall = correct_count / true_sum
            recall = np.round(recall, round_num) * per_num
        except ZeroDivisionError:
            recall = None
            
        # f1_score
        try:
            f1_score = 2*precision*recall / (precision + recall)
            f1_score = np.round(f1_score, round_num)
        except TypeError:
            f1_score = None
        
        matrix[i].extend([None, precision, recall, f1_score, true_sum])
        
    whole_accuracy = correct_sum / whole_sum
    whole_accuracy = np.round(whole_accuracy, round_num) * per_num
    
    # index & column
    index_list = uni_label_list.copy()
    index_list.append('count')
    column_list = uni_label_list.copy()
    column_list.extend(['accuracy', 'precision', 'recall', 'f1 score', 'count'])
    
    # count 추가
    pred_count = pred_sum_list + [None]*(len(column_list) - len(index_list))
    matrix.append(pred_count)
    
    # confusion matrix
    confusion_matrix = pd.DataFrame(matrix, index=index_list, columns=column_list)
    confusion_matrix['accuracy'][0] = whole_accuracy
    
    return confusion_matrix


def make_confusion_matrix(mode, true_list, pred_list, ignore_idx=None, round_num=4, label2id_dict=None, id2label_dict=None, show_percentage=True):
    if mode == 'label2id':
        uni_label_list = list(label2id_dict.keys())
    elif mode == 'id2label':
        uni_label_list = list(id2label_dict.values())
        
    # matrix
    matrix = []
    for i in range(len(uni_label_list)):
        matrix.append([])
        for _ in range(len(uni_label_list)):
            matrix[i].append(0)

    # count
    if mode == 'label2id':
        for t, p in zip(true_list, pred_list):
            t_i = label2id_dict[t]
            p_i = label2id_dict[p]
            matrix[t_i][p_i] += 1
            
    elif mode == 'id2label':
        for t_i, p_i in zip(true_list, pred_list):
            # padding 등의 idx 는 무시 
            if (t_i is not None) and (t_i == ignore_idx):
                continue
            t_i = int(t_i)
            p_i = int(p_i)
            matrix[t_i][p_i] += 1
            
    confusion_matrix = matrix2confusion(
        matrix=matrix, 
        uni_label_list=uni_label_list, 
        round_num=round_num, 
        show_percentage=show_percentage
    )
    
    return confusion_matrix


def reset_confusion_matrix(confusion_matrix, new_label_list, round_num=4, show_percentage=True):
    try:
        matrix_df = confusion_matrix[new_label_list]
    except KeyError:
        print('예측 결과에 존재하지 않는 라벨명을 입력하였습니다.')
        sys.exit()
        
    matrix_df = matrix_df.T[new_label_list]
    matrix_df = matrix_df.T
    matrix = matrix_df.values.tolist()
    
    new_confusion_matrix = matrix2confusion(
        matrix=matrix, 
        uni_label_list=new_label_list,
        round_num=round_num,
        show_percentage=show_percentage
    )
    
    return new_confusion_matrix