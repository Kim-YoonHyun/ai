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
    


def num_classification(data):
    raw_text = data['text']
    span_list = data['spans']
    
    # ==
    # raw_text = '접 수 번호 1 A A - 1110-111111: 제 계좌번호는 1002-362- 682257이고 대구은행 계좌번호는 508 -10-252713-3, 주민등록번호는 941008-1673914 입니다. 전번은 010 -6214- 3274, 010-1234-1547인데 잘 외우셨죠?'
    # raw_text = '안녕하세요 저는 김윤현입니다. 잘부탁드립니다.'
    # data['spans'] = []
    # span_list = data['spans']
    # ==
    
    coupler = r'[-.]'
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
        'p' : r'010'
    }
    
    # 일치하는 정규식의 길이 계산
    def extract_match(match):
        return f"§{len(match.group(0))}§"

    except_list = [1000]
    bank_list = [626, 436, 426, 365, 363, 3623, 3442, 336, 3261, 326, 3243]
    op_list = [325]
    regi_list = [67]
    phone_list = ['p44', 334, 244, 234, 44]
    
    temp_text = copy.deepcopy(raw_text)  # 길이 계산용 임시 텍스트
    print(temp_text)
    text = copy.deepcopy(raw_text)  # 원본 교체용 텍스트
    
    for nn, order_list in enumerate([except_list, bank_list, op_list, regi_list, phone_list]):
        
        for order in order_list:
            
            # 각 번호 규칙에 따라 진행
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
            temp_text = re.sub(regular_expression, extract_match, temp_text)
        
        # 특수문자 § 를 기준으로 스플릿
        text_split = temp_text.split('§')
        
        # 길이계산(span 계산용)
        text_len_split = list(map(get_len, text_split))
        
        # span 추가
        pre_end = 0
        num = 0
        new_text_split = []
        for t_l_s in text_len_split:
            try:
                # span 계산용으로 계속 더하기
                num += t_l_s
            except TypeError:
                # 정규식으로 추출한 부분에 도달한 경우
                start = num
                num += int(t_l_s)
                
                # 라벨 설정
                if nn == 0:
                    label = ''
                if nn == 1:
                    label = '계좌번호'
                if nn == 2:
                    label = '사업자등록번호'
                if nn == 3:
                    label = '주민등록번호'
                if nn == 4:
                    label = '전화번호'
                temp_span = {'start':start, 'end':num, 'label':label}
                
                # 접수번호인 경우 제외
                if nn != 0:
                    span_list.append(temp_span)
                
                # temp text 에서 이미 추출한 부분 재추출안되게 ㅁ으로 변환
                new_text_split.append(text[pre_end:start])
                new_text_split.append('|' * int(t_l_s))
                pre_end = num
        
        # 나머지 부분 append
        new_text_split.append(text[pre_end:])
        
        # 다시 붙이기
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