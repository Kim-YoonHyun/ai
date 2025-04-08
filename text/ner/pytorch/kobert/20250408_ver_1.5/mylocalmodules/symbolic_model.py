import sys
import os
import re
import copy
import numpy as np
import pandas as pd


def get_len(string):
    try:
        _ = int(string)
        return string
    except ValueError:
        return len(string)
    

# 일치하는 정규식의 길이 계산
def extract_match(match):
    len_list = []
    for m in match.group().split(' '):
        len_list.append(f"§{len(m)}§")
    result = ' '.join(len_list)
    # return f"§{len(match.group(0))}§"
    return result


def num_classification(data, num_list_dict):
    raw_text = data['text']
    span_list = data['spans']
    # ==
    # raw_text = '접 수 번호 1 A A - 1110-111111: 제 계좌번호는 1002-362- 682257이고 대구은행 계좌번호는 508 -10-252713-3, 주민등록번호는 941008-1673914 입니다. 전번은 010 -6214- 3274, 010-1234-1547인데 잘 외우셨죠?'
    # raw_text = '안녕하세요 저는 김윤현입니다. 사업자 등록 번호 111-11 - 11111 전화 번호 010-1111 - 1111'
    # data['spans'] = []
    # span_list = data['spans']
    # ==
    
    coupler = r'[-.\s]'
    blank = r'\s?'
    connector = f'{blank}{coupler}{blank}'
    num_dict = {
        '1' : r'\d{1}',
        '2' : r'\d{2}',
        '3' : r'\d{3}',
        '4' : r'\d{4}',
        '5' : r'\d{5}',
        '6' : r'\d{6}',
        '7' : r'\d{7}',
        '8' : r'\d{8}',
        '9' : r'\d{9}'
    }
    
    num_label_list = list(num_list_dict.keys())
    num_order_list = list(num_list_dict.values())
    
    temp_text = copy.deepcopy(raw_text)  # 길이 계산용 임시 텍스트
    text = copy.deepcopy(raw_text)  # 원본 교체용 텍스트
    
    for num_label, num_order in zip(num_label_list, num_order_list):
        num_order.sort(reverse=True)
        
        # 규칙에 따른 번호 추출
        for order in num_order:
            
            # 각 번호별 정규식 생성
            if order == 1000:
                # 접수번호인 경우
                regular_expression = f"{num_dict['1']}{blank}[a-zA-Z]{blank}[a-zA-Z]{connector}{num_dict['4']}{connector}{num_dict['6']}"
            else:
                # 그외 번호인 경우
                order_split = '.'.join(str(order)).split('.')
                
                regular_expression = r'\b'
                lim_num = len(order_split) - 2
                for i, o_s in enumerate(order_split):
                    num_reg = num_dict[o_s]
                    regular_expression += num_reg
                    if i <= lim_num:
                        regular_expression += connector
                # regular_expression += r'\b'
            
            # 정규식에 해당되는 부분을 § 로 구분하여 길이값으로 변환 추출
            temp_text = re.sub(regular_expression, extract_match, temp_text)
            # ==
            # print(num_label, temp_text)
            # ==
        
        # 특수문자 § 를 기준으로 스플릿
        text_split = temp_text.split('§')
        
        # 길이계산(span 계산용)
        # 번호 길이는 str, 그외 부분 길이는 int 값으로 반환
        text_len_split = list(map(get_len, text_split))
        
        # span 추가
        pre_end = 0
        num = 0
        new_text_split = []
        # print(text_len_split)
        for t_l_s in text_len_split:
            try:
                # span 계산용으로 계속 더하기
                num += t_l_s
            except TypeError:
                # 정규식으로 추출한 부분(번호)에 도달한 경우
                start = num
                num += int(t_l_s)
                
                # 라벨 설정
                label = f'B▲{num_label}'
                temp_span = {'start':start, 'end':num, 'label':label}
                
                # 제외리스트는 제외
                if num_label != "except":
                    span_list.append(temp_span)
                
                # 기존의 text 에서 추출된 부분의 앞 문장을 저장
                new_text_split.append(text[pre_end:start])
                # temp text 에서 이미 추출한 부분 재추출안되게 |으로 변환
                new_text_split.append('|' * int(t_l_s))
                pre_end = num
        
        # 나머지 뒷 부분 append
        new_text_split.append(text[pre_end:])
        
        # 번호가 추출된 text 생성
        temp_text = ''.join(new_text_split)
        text = temp_text
    
    # 기존의 span 도 추출된 번호도 없는 경우
    if len(span_list) == 0:
        return data
    
    # span_list 정렬
    sort_span_list = []
    span_df = pd.DataFrame(span_list)
    span_df = span_df.sort_values(by='start')
    for _, row in span_df.iterrows():
        start = row[0]
        end = row[1]
        label = row[2]
        temp_span = {'start':start, 'end':end, 'label':label}
        sort_span_list.append(temp_span)
    
    data['spans'] = sort_span_list
    return data


def symbolic_ai(data_list, num_list_dict):
    new_data_list = []
    for data in data_list:
        new_data = num_classification(
            data=data, 
            num_list_dict=num_list_dict
        )
        new_data_list.append(new_data)
    return new_data_list
