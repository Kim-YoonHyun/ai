import re
import sys
import json

import pandas as pd

from tqdm import tqdm
from pprint import pprint
from konlpy.tag import Okt


def is_korean(s):
    for char in s:
        if not ('\uAC00' <= char <= '\uD7A3'):
            return False
    return True


def temp_space(token_list):
    temp = []
    First = True
    for num in range(len(token_list)):
        if First:
            temp.append(token_list[num])
            First = False
        elif len(token_list[num]) == 1:
            temp.append(token_list[num])
        else:
            if is_korean(token_list[num]):
                pos_pre = MORPH.pos(token_list[num-1])
                pos_post = MORPH.pos(token_list[num])
                if  (len(pos_pre) == 0) or (len(pos_post)==0):
                    temp.append(token_list[num])
                elif (pos_pre[0][1] == "Josa") and (pos_post[0][1] == "Noun"):
                    temp.append("_"+token_list[num])
                else:
                    temp.append(token_list[num])
            else:
                temp.append(token_list[num])
    return temp


class CustomRule():
    def __init__(self):
        self.final_ending = {
            "4" : [
                "있습니다", "었습니다", "더라구요", "했습니다", "였습니다", "졌습니다", "주십시요", "겠습니다", "주십시오", "습니다만", "셨습니다", "겠습니까", "주셨으면", "더라고요", 
                "야하는데", "싶습니다", "것이라고", "다는것은", "하더라도", "하겠지만", "수있도록", "왔습니다", "야하는지", "주겠다고", "하였으나", "갔습니다", "수있다고", "야하나요", 
                "웠습니다", "주었으면", "야하는게", "다고하고", "하십니다", "는것으로", "았습니다", "주겠다는", "있음에도", "것이라는", "하더군요", "하면서도", "한것으로", "다는것을", 
                "놨습니다", "하였으며", "하였지만", "있었는데", "다니면서", "고있는데", "다는것이", "수있다는", "어주셔서"
            ],
            "3" : [
                "습니다", "합니다", "주세요", "하다고", "습니까", "더군요", "하지만", "하다는", "것으로", "는데도", "는데요", "하거나", "더라도", "것이다", "음에도", "있는데", "는것이", 
                "다면서", "는것은", "겠다고", "주시길", "다는데", "겠지만", "겠다는", "하면서", "는것도", "거라고", "주시기", "주시고", "하는데", "라구요", "었는데", "가면서", "야하고", 
                "답니다", "다라고", "주시면", "수있는", "있다고", "것이고", "하네요", "해지고", "지면서", "다는게", "있어서", "십시요", "는것을", "면서도", "겠지요", "하였고", "준다고"
            ],
            "2" : [
                "는데", "다고", "하고", "니다", "다는", "하게", "지만", "면서", "거나", "하여", "라고", "다면", "니까", "는지", "지는", "려고", "네요", "는게", "다가", "지도", "으면", 
                "라는", "하다", "기에", "하지", "있는", "지요", "어요", "지고", "해서", "하는", "하며", "것이", "으나", "으며", "것도", "도록", "고요", "냐고", "하면", "나요", "가는", 
                "더니", "것은", "으니", "으로", "하기", "것을", "는건", "다며"
            ]
        }
        with open("./pattern.json", "r") as f: PATTERN_DATA = json.load(f)
        self.pattern_list = PATTERN_DATA["pattern"]
        self.MORPH = Okt()

    def post_inner_ending(self, pos_tuple, end_size):
        if len(pos_tuple[0]) == end_size:
            token = [pos_tuple[0]]
            is_number = [0]
            need_space = [0]
        else:
            token, is_number, need_space= [], [], []
            for word in [pos_tuple[0][:-end_size], pos_tuple[0][-end_size:]]:
                token.append(word)
                is_number.append(0)
                need_space.append(0)
        return token, is_number, need_space  

    def post_ending(self, pos_tuples, pos_num):
        if pos_tuples[pos_num][0][-4:] in self.final_ending[str(4)]:
            token, is_number, need_space = self.post_inner_ending(pos_tuples[pos_num], 4)
        elif pos_tuples[pos_num][0][-3:] in self.final_ending[str(3)]:
            token, is_number, need_space = self.post_inner_ending(pos_tuples[pos_num], 3)
        elif pos_tuples[pos_num][0][-2:] in self.final_ending[str(2)]:
            token, is_number, need_space = self.post_inner_ending(pos_tuples[pos_num], 2)
        else:
            if len(pos_tuples[pos_num][0]) > 3:
                token, is_number, need_space= [], [], []
                token.append(pos_tuples[pos_num][0][:3])
                is_number.append(0)
                need_space.append(0)
                for alpha in pos_tuples[pos_num][0][3:]:
                    token.append(alpha)
                    is_number.append(0)
                    need_space.append(0)
            else:
                token = [pos_tuples[pos_num][0]]
                is_number = [0]
                need_space = [0]
        return token, is_number, need_space
    
    def noun_inner_rule(self, pos_num, pos_tuples):
        if pos_tuples[pos_num][0] in self.pattern_list:
            token = [pos_tuples[pos_num][0]]
            is_number = [0]
            need_space = [0]

        else:
            token = []
            is_number = []
            need_space = []                
            for alpha in pos_tuples[pos_num][0]:
                token.append(alpha)
                is_number.append(0)
                need_space.append(0)
        return token, is_number, need_space    

    def noun_rule(self, pos_tuples, pos_num):
        if pos_num == 0:
            token, is_number, need_space = self.noun_inner_rule(pos_num, pos_tuples) 
        else:
            if pos_tuples[pos_num-1][1] == "Josa":
                if pos_tuples[pos_num][0] in self.pattern_list:
                    token = [pos_tuples[pos_num][0]]
                    is_number = [0]
                    need_space = [1]
                else:
                    token, is_number, need_space= [], [], []            
                    for num, alpha in enumerate(pos_tuples[pos_num][0]):
                        token.append(alpha)
                        is_number.append(0)
                        if num == 0:
                            need_space.append(1)
                        else:
                            need_space.append(0)                            
            else:
                token, is_number, need_space = self.noun_inner_rule(pos_num, pos_tuples)
        return token, is_number, need_space

    def hashtag_rule(self, pos_tuples, pos_num):
        token, is_number, need_space= [], [], []
        for num, word in enumerate(["#", pos_tuples[pos_num][0][1:]]):
            if num == 0:
                token.append(word)
                is_number.append(0)
                need_space.append(0)
                
            else:
                inner_pos_tuples = self.MORPH.pos(word)
                for num, inner_pos in enumerate(inner_pos_tuples):
                    if (inner_pos[1]=="Noun") and (inner_pos[0] in self.pattern_list):
                        token.append(inner_pos[0])
                        is_number.append(0)
                        if (num==0) and (inner_pos[1]=="Noun"):
                            need_space.append(1)
                        else:
                            need_space.append(0)
                    else:
                        if inner_pos[1] == "Number":
                            origin_number = inner_pos[0]
                            modify_number = "".join(['1' if char.isdigit() and char != '0' else char for char in origin_number])
                            token.append(modify_number)
                            is_number.append(0)
                            need_space.append(0)
                        else:
                            for alpha in inner_pos[0]:
                                token.append(alpha)
                                is_number.append(0)
                                need_space.append(0)
        return token, is_number, need_space

    def number_rule(self, pos_tuples, pos_num):
        origin_number = pos_tuples[pos_num][0]
        modify_number = "".join([
            '1' if char.isdigit() and char != '0' else char for char in origin_number
            ])
        token = [modify_number]
        is_number = [origin_number]
        need_space = [0]
        return token, is_number, need_space

    def post_inner_ending(self, pos_tuple, end_size):
        if len(pos_tuple[0]) == end_size:
            token = [pos_tuple[0]]
            is_number = [0]
            need_space = [0]
        else:
            token, is_number, need_space= [], [], []
            for word in [pos_tuple[0][:-end_size], pos_tuple[0][-end_size:]]:
                token.append(word)
                is_number.append(0)
                need_space.append(0)
        return token, is_number, need_space  
    
    def etc_rule(self, pos_tuples, pos_num):
        token, is_number, need_space= [], [], []
        if is_korean(pos_tuples[pos_num][0]):
            inner_pos_tuples = self.MORPH.pos(pos_tuples[pos_num][0])
            for num, inner_pos in enumerate(inner_pos_tuples):
                if (inner_pos[1]=="Noun") and (inner_pos[0] in self.pattern_list):
                    token.append(inner_pos[0])
                    is_number.append(0)
                    need_space.append(0)
                elif (inner_pos[1]=="Verb") or (inner_pos[1]=="Adjective"):
                    if inner_pos[0][-4:] in self.final_ending[str(4)]:
                        _token, _is_number, _need_space = self.post_inner_ending(inner_pos, 4)
                    elif inner_pos[0][-3:] in self.final_ending[str(3)]:
                        _token, _is_number, _need_space = self.post_inner_ending(inner_pos, 3)
                    elif inner_pos[0][-2:] in self.final_ending[str(2)]:
                        _token, _is_number, _need_space = self.post_inner_ending(inner_pos, 2)
                    else:
                        _token = [inner_pos[0]]
                        _is_number = [0]
                        _need_space = [0]
                    token.extend(_token)
                    is_number.extend(_is_number)
                    need_space.extend(_need_space)         
                else:
                    for alpha in inner_pos[0]:
                        token.append(alpha)
                        is_number.append(0)
                        need_space.append(0)
        else:
            for alpha in pos_tuples[pos_num][0]:
                token.append(alpha)
                is_number.append(0)
                need_space.append(0)
        return token, is_number, need_space
    
    def particle_rule(self, pos_tuples, pos_num):
        token, is_number, need_space= [], [], []
        for alpha in pos_tuples[pos_num][0]:
            token.append(alpha)
            is_number.append(0)
            need_space.append(0)
        return token, is_number, need_space

MORPH = Okt()
CUSTOM = CustomRule()

def apply_rule(
        pos_tuples, pos_num, token_list, number_position, space_position
               ):
    if pos_tuples[pos_num][1] == "Noun":
        token, is_number, need_space  = CUSTOM.noun_rule(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] == "Verb":
        token, is_number, need_space = CUSTOM.post_ending(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] == "Adjective":
        token, is_number, need_space = CUSTOM.post_ending(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] == "Hashtag":
        token, is_number, need_space = CUSTOM.hashtag_rule(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] == "Number":
        token, is_number, need_space = CUSTOM.number_rule(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] in ["Foreign", 'URL', 'Email', "Punctuation", "Alpha"]:
        token, is_number, need_space = CUSTOM.etc_rule(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    elif pos_tuples[pos_num][1] == "KoreanParticle":
        token, is_number, need_space = CUSTOM.particle_rule(
            pos_tuples, pos_num)
        token_list.extend(token)
        number_position.extend(is_number)
        space_position.extend(need_space)

    else:
        token_list.append(pos_tuples[pos_num][0])
        number_position.append(0)
        space_position.append(0) 
    return token_list, number_position, space_position

def okt_base_tokenizer(sentence):
    """
    okt를 이용하여 몇가지 룰을 적용한 후 만든 토크나이져입니다.

    Arg:
        sentence (str) : 원문

    Returns:
        token_list (list) : 토큰화된 리스트
        number_position (list) : 숫자 변환이 일어난 부분과 숫자 원문을 포함한 리스느 0 은 숫자 분분 아님
        space_position (list) : 명사앞에 조사가 사용된 부분으로 토큰에 "_"가 추가 되야 하는 부분(미완성)
        sentence (str) : 원문

    Examples
        sentence = "더아이엠씨는 www.theimc.co.kr와 053-777-8373로 확인 가능."

        token_list, number_position, space_position, sentence = okt_base_tokenizer(origin_sentence)

        print(token_list)
        print(number_position)
        print(space_position)
        print(sentence)

        # out : ['더', '아이엠', '씨', '는', '직원', 'w', 'w', 'w', '.', 't', 'h', 'e', 'i', 'm', 'c', '.', 'c', 'o', '.', 'k', 'r', '와', '011-111', '-', '1111', '로', '확', '인', '가', '능', '.']
        # out : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '053-777', 0, '8373', 0, 0, 0, 0, 0, 0]
        # out : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 더아이엠씨는 직원 www.theimc.co.kr와 053-777-8373로 확인 가능.
    """
    # okt 형태도 분석기 적용
    pos_tuples = MORPH.pos(sentence)
    number_position = []
    space_position = []
    token_list = []

    for pos_num in range(len(pos_tuples)):
        if len(pos_tuples[pos_num][0])<=8:
            token_list, number_position, space_position = apply_rule(
                pos_tuples, pos_num, token_list, number_position, space_position
                )

        else:
            inner_pos_tuples = MORPH.pos(pos_tuples[pos_num][0])
            for inner_pos_num in range(len(inner_pos_tuples)):
                if len(inner_pos_tuples[inner_pos_num]) > 8:
                    print(inner_pos_tuples[inner_pos_num])
                token_list, number_position, space_position = apply_rule(
                    inner_pos_tuples, inner_pos_num, token_list, number_position, space_position
                    )

    return token_list, number_position, space_position, sentence


def restoraion(token_list, number_position, sentence):

    token_temp_list = []
    for number, token in zip(number_position, token_list):
        if number !=0:
            token_temp_list.append(number)
        else:
            token_temp_list.append(token)
    origin_space = []
    for token in token_temp_list:
        if sentence[0] == " ":
            origin_space.append(1)
            sentence=sentence[1:]
        else:
            origin_space.append(0)
        for to in token:
            if sentence[0] == to:
                sentence=sentence[1:]
    return origin_space








if __name__ == "__main__" :

    origin_sentence_list = [
        "일어났는지를 VILLAGE 더아이엠씨는 직원 www.theimc.co.kr와 053-777-8373로 확인 가능."
    ]


    modify_dict = []
    n =0

    for origin_sentence in origin_sentence_list: # tqdm(origin_sentence_list):
        if n%13==0:
            print(n)
        ###  hj_tokenizer 적용 부분
            token_list, number_position, space_position, sentence = okt_base_tokenizer(
                origin_sentence)
            origin_space = restoraion(token_list, number_position, sentence)
            restore_sentence = ""
            for num, token in zip(origin_space, token_list):
                if num == 1:
                    restore_sentence  += " " + token
                else:
                    restore_sentence += token

            
            modify_list = {
                "text" : " ".join(token_list), 
                "index" : n, 
                "number_position" : number_position, 
                "underbar_position" : space_position,
                "origin_space" : origin_space
                }

            modify_dict.append(modify_list)
        n+=1
