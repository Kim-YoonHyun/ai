import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch

from transformers import BertTokenizer
from torch.autograd import Variable
from mylocalmodules import transformer
from mylocalmodules import dataloader as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm

def subsequent_mask(size):
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
                                                                    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path')
    parser.add_argument('--trained_model_name')

    # train variable
    parser.add_argument('--device_num', type=int, default=0)
    
    # dataloader variable    
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)

    args = parser.parse_args()
    root_path = args.root_path
    trained_model_path = args.trained_model_path
    trained_model_name = args.trained_model_name
    with open(f'{trained_model_path}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        trained_args_setting = json.load(f)
    
    device_num = trained_args_setting['device_num']
    num_embeddings = trained_args_setting['num_embeddings']
    d_model = trained_args_setting['d_model']
    head_num = trained_args_setting['head_num']
    max_length = trained_args_setting['max_length']
    dropout_p = trained_args_setting['dropout_p']
    layer_num = trained_args_setting['layer_num']
    
    # ===========================================================================    
    device = trm.get_device(device_num)
    
    # ===========================================================================    
    model = transformer.Transformer(
        num_embeddings=num_embeddings, 
        d_model=d_model, 
        max_seq_len=max_length, 
        head_num=head_num, 
        dropout_p=dropout_p, 
        layer_num=layer_num
    )
    weight = torch.load(f'{trained_model_path}/{trained_model_name}/weight.pt')
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    
    # ===========================================================================    
    input_list = []
    label_list = []
    data_name_list = os.listdir(f'{root_path}/datasets/ttt')
    data_name_list.sort()
    for data_name in data_name_list:
        df = pd.read_csv(f'{root_path}/datasets/ttt/{data_name}', encoding='utf-8-sig')
        ECU_VehicleSpeed_STD = [num_embeddings-2] + df['ECU_VehicleSpeed_STD'].values.tolist()
        ECU_EngineSpeed = [num_embeddings-2] + df['ECU_EngineSpeed'].values.tolist()
        input_list.append(ECU_VehicleSpeed_STD)
        label_list.append(ECU_EngineSpeed)
    
    Test_Dataloader = dam.get_dataloader(
        input_list=input_list[-5:],
        label_list=label_list[-5:],
        num_embeddings=num_embeddings,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=5,
        pin_memory=True
    )
    
    # for x, x_mask, target, target_mask in Val_Dataloader:
    #     print(x.size())
    
    # string = tokenizer.encode(input_str)
    # string_len = len(string)
    # pad_len = (max_length - string_len)
    # encoder_input = torch.tensor(string + [tokenizer.pad_token_id]*pad_len)
    # encoder_mask = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)
    
    
    
    for x, x_mask, yyy, _ in Test_Dataloader:
        x = x.to(device, dtype=torch.int)
        while len(x_mask.size()) < 4:
            x_mask = x_mask.unsqueeze(1)
        x_mask = x_mask.to(device, dtype=torch.float)
        
        
        b_label = torch.ones(1, 1).fill_(149).type_as(x)
        # b_label = b_label.to(device).long()
        # while len(b_label_mask.size()) < 4:
        #     b_label_mask = b_label_mask.unsqueeze(1)
        # b_label_mask = b_label_mask.to(device, dtype=torch.float)
        
        encoder_output = model.encode(x, x_mask)
        
        for i in range(max_length - 1):
            b_label_mask = Variable(subsequent_mask(b_label.size(1)).type_as(x.data))
            pred = model.decode(
                encode_output=encoder_output,
                encoder_mask=x_mask,
                target=b_label,
                target_mask=b_label_mask
            )
            prob = pred[:, -1]
            
            _, next_value = torch.max(prob, dim=1)
            
            # sys.exit()
            # print(input_str)
            # output_str = tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)
            b_label = torch.cat((b_label[0], next_value))
            b_label = b_label.unsqueeze(0)
            
            # if i > 10:
            #     sys.exit()
        # print(b_label)
        print(yyy[0][:50])
        print(b_label[0][:50])
        print(b_label.size())
        sys.exit()
            
        
    
    encoder_output = model.encode(encoder_input, encoder_mask)
    for i in range(max_length - 1):
        target_mask = Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data))
        # print(target_mask)
        # target_mask = Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data))
        # print(target_mask)
        pred = model.decode(
            encode_output=encoder_output,
            encoder_mask=encoder_mask,
            target=target,
            target_mask=target_mask
        )
        # print(encoder_mask)
        # print(pred)
        # print(pred.size())
        prob = pred[:, -1]
        # print(prob)
        # print(prob.size())
        # sys.exit()
        
        _, next_word = torch.max(prob, dim=1)
        
        # sys.exit()
        # print(input_str)
        output_str = tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)
        target = torch.cat((target[0], next_word))
        target = target.unsqueeze(0)
        
        # sys.exit()

if __name__ == '__main__':
    main()