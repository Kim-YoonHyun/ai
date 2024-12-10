import sys
import os
sys.path.append(os.getcwd())
import json
import copy
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch

from mylocalmodules import dataloader as dam
from model import network as net
import model_trainer as mt

# share modules
sys.path.append('/home/kimyh/python/ai')
from sharemodule import logutils as lom
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import utils as utm

# from transformers import AutoTokenizer
# from konlpy.tag import Okt
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer



def tester():
    root_path = '/home/kimyh/python/project/transformer'
    device_num = 0
    
    # train parameter
    dropout_p = 0.1
    random_seed = 42
    
    # network parameter
    # max_len = 12
    # d_model = 8
    # d_ff = d_model*4
    # n_heads = 4
    
    max_len = 128
    d_model = 256
    d_ff = 512
    n_heads = 8
    
    # layer num parameter
    enc_layer_num = 6
    dec_layer_num = 6
    
    test_sentence = '앞으로 좋은 일만 있었으면 좋겠어'
    
    # =========================================================================
    # seed 지정
    utm.envs_setting(random_seed)
    
    # =========================================================================
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    
    kor_ids = tokenizer(
        test_sentence, 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
    ).input_ids
    print(kor_ids)
    # ===================================================================    
    # device 지정
    device = tum.get_device(device_num)
    
    # =========================================================================
    # model 생성
    model = net.Transformer(
        device=device,
        max_len=max_len,
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        pad_idx=pad_idx,
        enc_layer_num=enc_layer_num,
        dec_layer_num=dec_layer_num,
        dropout_p=dropout_p
    )
    
    # 학습된 가중치 로딩
    weight_path = f"{root_path}/trained_model/chat/epoch0020/weight.pt"
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    
    # 평가 모드 전환
    model.eval()
    
    # ==================================================================
    # 사전 학습된 모델로 번역해보기
    input_text = "앞으로 해야하는 강의를 어떻게 진행할 것인지 고민해본다."
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    pre_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    pre_model.to(device)
    pre_model.eval()
    translated_tokens = pre_model.generate(input_tokens, max_new_tokens=max_len)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    # ==================================================================
    # encoder 입력
    with torch.no_grad():
        x_input = torch.tensor(kor_ids).unsqueeze(0).to(device)#, dtype=torch.float)
        enc_self_mask, _ = dam.get_self_mask(x_input, n_heads, pad_idx)
        enc_self_mask = enc_self_mask.to(device)
        enc_out = model.encoder(x_input, enc_self_mask)
    
    # =========================================================================
    # decoder 입력
    eng_ids = [eos_idx]
    
    for i in range(max_len):
        # print('===================================')
        # mask 생성
        
        
        with torch.no_grad():
            y_input = torch.tensor(eng_ids).unsqueeze(0).to(device)
            _, y_la_mask = dam.get_self_mask(y_input, n_heads, pad_idx)
            y_la_mask = y_la_mask.to(device)
            enc_dec_mask = dam.get_enc_dec_mask(x_input, y_input, n_heads, pad_idx)
            enc_dec_mask = enc_dec_mask.to(device)
            
            output = model.decoder(y_input, enc_out, y_la_mask, enc_dec_mask)
        
            # 선택
            pred = output.argmax(2)
            # sys.exit()
            pred_id = pred[:, -1].item()
            # print(pred[:, -1].size())
            eng_ids.append(pred_id)
            # sys.exit()
        
        if pred_id == eos_idx:
            break
        
    # ==================================================================
    # index 를 token 으로 변환
    # result_tokens = []
    # for pred_id in eng_ids:
    #     if pred_id == 0:
    #         continue
    #     # pred_id = str(pred_id)
    #     result_tokens.append(tokenizer.decode(pred_id))
    # result_sentence = ' '.join(result_tokens[1:-1])
    result_sentence = tokenizer.decode(eng_ids)
    print(test_sentence)
    print(result_sentence)
        
    sys.exit()
        
    
def main():
    # args = get_args()    
    # test_args_setting = vars(args)
    
    # # =========================================================================
    # # log 생성
    # log = lom.get_logger(
    #     get='TEST',
    #     root_path=test_args_setting['root_path'],
    #     log_file_name=f'test.log',
    #     time_handler=True
    # )
    
    # =========================================================================
    # 평가 진행
    tester()

        
if __name__  == '__main__':
    main()
