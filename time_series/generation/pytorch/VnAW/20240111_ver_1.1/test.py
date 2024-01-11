import sys
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# from transformers import BertTokenizer
from torch.autograd import Variable
from models_02 import VnAW as net
from mylocalmodules import dataloader as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import train as trm
from sharemodule import trainutils as tum


def subsequent_mask(size):
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def concat(x):
    x_0 = str(int(x[0]))
    x_1 = str(int(x[1]))
    new_x = int(x_0 + x_1)
    return new_x
                                                     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    parser.add_argument('--trained_model_path1')
    parser.add_argument('--trained_model_path2')
    parser.add_argument('--trained_model_name')
    parser.add_argument('--test_data_path')

    # dataloader variable    
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)

    args = parser.parse_args()
    root_path = args.root_path
    
    trained_model_path1 = args.trained_model_path1
    trained_model_path2 = args.trained_model_path2
    trained_model_name = args.trained_model_name
    test_data_path = args.test_data_path
    with open(f'{trained_model_path1}/args_setting.json', 'r', encoding='utf-8-sig') as f:
        trained_args_setting = json.load(f)
    
    device_num = trained_args_setting['device_num']
    d_model = trained_args_setting['d_model']
    head_num = trained_args_setting['head_num']
    max_length = trained_args_setting['max_length']
    dropout_p = trained_args_setting['dropout_p']
    layer_num = trained_args_setting['layer_num']
    
    # ===========================================================================    
    device = tum.get_device(device_num)
    
    # ===========================================================================    
    model = net.VnAW(
        d_model=d_model, 
        max_length=max_length, 
        head_num=head_num, 
        dropout_p=dropout_p, 
        layer_num=layer_num
    )
    weight = torch.load(f'{trained_model_path1}/{trained_model_path2}/{trained_model_name}/weight.pt')
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    
    # ===========================================================================    
    test_data_list = os.listdir(test_data_path)
    test_data_list.sort()
    
    for test_data in test_data_list:
        # 개별 입력
        Test_Dataloader = dam.get_dataloader(
            dataset_path=test_data_path, 
            data_name_list=[test_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True
        )
        
        for x, x_mask, b_label in tqdm(Test_Dataloader):
            # x
            x = x.to(device, dtype=torch.float32)
            
            # x mask
            while len(x_mask.size()) < 4:
                x_mask = x_mask.unsqueeze(1)
            x_mask = x_mask.to(device, dtype=torch.float32)
            
            # b label
            b_label = b_label.to(device, dtype=torch.float32)
            
            # # b label mask
            # while len(b_label_mask.size()) < 4:
            #     b_label_mask = b_label_mask.unsqueeze(1)
            # b_label_mask = b_label_mask.to(device)

            # output
            pred = model(
                input=x,
                input_mask=x_mask
            )
            output_ary = pred.squeeze().cpu().detach().numpy()
            output_ary = output_ary * 100
            
            # figure
            true_ary = b_label.squeeze().cpu().detach().numpy()
            true_ary = true_ary * 100
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(30, 5))
            plt.xlim(0, 600)
            # plt.ylim(27, 29)
            
            plt.title('true')
            plt.plot(true_ary, lw=3, c='b')
            plt.plot(output_ary, lw=3, c='g')
            
            # plt.savefig(f'{root_path}/true_pred.png')
            plt.savefig(f'./true_pred.png')
            plt.clf()
            
            # plt.figure(figsize=(20, 5))
            # plt.title('pred')
            
            # plt.xlim(0, 100)
            
            # plt.savefig(f'{root_path}/pred.png')
            # plt.clf()
            sys.exit()
        
    # # input_list = []
    # # label_list = []
    # # data_name_list = os.listdir(f'{root_path}/datasets/{dataset_name}')
    # # data_name_list.sort()
    # # for data_name in data_name_list:
    # #     df = pd.read_csv(f'{root_path}/datasets/{dataset_name}/{data_name}', encoding='utf-8-sig')
    # #     ECU_VehicleSpeed_STD_ary = np.expand_dims(df['ECU_VehicleSpeed_STD'].values, axis=-1)
    # #     ECU_EngineSpeed_ary = np.expand_dims(df['ECU_EngineSpeed'].values * 0.1, axis=-1)
    # #     speed_ary = np.concatenate((ECU_EngineSpeed_ary, ECU_VehicleSpeed_STD_ary), axis=-1)
    # #     speed_ary = [2] + list(map(concat, speed_ary)) + [3]
    # #     input_list.append(speed_ary)
        
    # #     ECU_BatteryVolt_list = [2] + list(map(int, df['ECU_CoolantTemp'].values * 10)) + [3]
    # #     label_list.append(ECU_BatteryVolt_list)
        
    
    # #==
    # # input_list = []
    # # label_list = []
    # # with open(f'{root_path}/datasets/colloquial_00.json', encoding='utf-8-sig') as f:
    # #     data_dict = json.load(f)
    # # kor_list = data_dict['validation']['kor_list'][:100]
    # # en_list = data_dict['validation']['eng_list'][:100]
    # # vocab_path = f'{root_path}/vocab/wiki-vocab.txt'
    # # tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    # # padding_id = tokenizer.pad_token_id
    # # for kor, en in zip(tqdm(kor_list), en_list):
    # #     kor_ids_list = tokenizer.encode(kor, max_length=max_length, truncation=True)
    # #     rest = max_length - len(kor_ids_list)
    # #     kor_ids_pad_list = kor_ids_list + [padding_id]*rest
    # #     kor_token_list = tokenizer.convert_ids_to_tokens(kor_ids_pad_list)
    # #     input_list.append(kor_ids_pad_list)
        
    # #     en_ids_list = tokenizer.encode(en, max_length=max_length, truncation=True)
    # #     rest = max_length - len(en_ids_list)
    # #     en_ids_pad_list = en_ids_list + [padding_id]*rest
    # #     en_token_list = tokenizer.convert_ids_to_tokens(en_ids_pad_list)
    # #     label_list.append(en_ids_pad_list)
    # #==
    
    # # Test_Dataloader = dam.get_dataloader(
    # #     input_list=input_list[-5:],
    # #     label_list=label_list[-5:],
    # #     num_embeddings=num_embeddings,
    # #     batch_size=1,
    # #     shuffle=False,
    # #     drop_last=False,
    # #     num_workers=5,
    # #     pin_memory=True
    # # )
    
    # # for x, x_mask, target, target_mask in Val_Dataloader:
    # #     print(x.size())
    
    # # string = tokenizer.encode(input_str)
    # # string_len = len(string)
    # # pad_len = (max_length - string_len)
    # # encoder_input = torch.tensor(string + [tokenizer.pad_token_id]*pad_len)
    # # encoder_mask = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)
    
    
    
    # for x, x_mask, b_label, _ in Test_Dataloader:
    #     x = x.to(device, dtype=torch.float32)
    #     # while len(x_mask.size()) < 4:
    #         # x_mask = x_mask.unsqueeze(1)
    #     x_mask = x_mask.to(device, dtype=torch.float32)
    #     print(x.size())
    #     print(x_mask.size())
    #     sys.exit()
        
        
    #     test_output = torch.ones(1, 1).fill_(2).type_as(x)
        
    #     encoder_output = model.encode(x, x_mask)
        
    #     for i in range(max_length - 1):
    #         test_output_mask = Variable(subsequent_mask(test_output.size(1)).type_as(x.data))
    #         pred = model.decode(
    #             encode_output=encoder_output,
    #             encoder_mask=x_mask,
    #             target=test_output,
    #             target_mask=test_output_mask
    #         )
    #         prob = pred[:, -1]
            
    #         _, next_value = torch.max(prob, dim=1)
            
    #         # sys.exit()
    #         # print(input_str)
    #         # output_str = tokenizer.decode(b_label.squeeze().tolist(), skip_special_tokens=True)
    #         test_output = torch.cat((test_output[0], next_value))
    #         test_output = test_output.unsqueeze(0)
            
    #         # if i > 10:
    #         #     sys.exit()
            
    #     true_ary = b_label.cpu().numpy()[0][1:-1] * 0.1
    #     true_list = list(map(int, true_ary))
    #     pred_ary = test_output.cpu().numpy()[0][1:-1] * 0.1
    #     pred_list = list(map(int, pred_ary))
        
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(20, 5))
    #     plt.title('true')
    #     plt.plot(true_list)
    #     plt.xlim(0, 600)
    #     # plt.ylim(26, 30)
    #     plt.savefig('./true.png')
    #     plt.clf()
        
    #     plt.figure(figsize=(20, 5))
    #     plt.title('pred')
    #     plt.plot(pred_list)
    #     plt.xlim(0, 600)
    #     # plt.ylim(26, 30)
    #     plt.savefig('./pred.png')
    #     plt.clf()
    #     sys.exit()
            
        
    
    # encoder_output = model.encode(encoder_input, encoder_mask)
    # for i in range(max_length - 1):
    #     target_mask = Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data))
    #     # print(target_mask)
    #     # target_mask = Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data))
    #     # print(target_mask)
    #     pred = model.decode(
    #         encode_output=encoder_output,
    #         encoder_mask=encoder_mask,
    #         target=target,
    #         target_mask=target_mask
    #     )
    #     # print(encoder_mask)
    #     # print(pred)
    #     # print(pred.size())
    #     prob = pred[:, -1]
    #     # print(prob)
    #     # print(prob.size())
    #     # sys.exit()
        
    #     _, next_word = torch.max(prob, dim=1)
        
    #     # sys.exit()
    #     # print(input_str)
    #     output_str = tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)
    #     target = torch.cat((target[0], next_word))
    #     target = target.unsqueeze(0)
        
        # sys.exit()

if __name__ == '__main__':
    main()