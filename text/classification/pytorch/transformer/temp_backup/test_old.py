import sys
import os
import numpy as np
import torch

from transformers import BertTokenizer
from torch.autograd import Variable
from mylocalmodules import transformer


def subsequent_mask(size):
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
                                                                    

def main():
    root_path = '/home/kimyh/python/project/transformer'
    vocab_path = f'{root_path}/vocab/wiki-vocab.txt'
    num_embeddings = 25000
    d_model = 16
    max_length = 15
    head_num = 8
    dropout_p = 0.2
    layer_num = 6
    
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    
    model = transformer.Transformer(
        num_embeddings=num_embeddings, 
        d_model=d_model, 
        max_seq_len=max_length, 
        head_num=head_num, 
        dropout_p=dropout_p, 
        layer_num=layer_num
    )
    checkpoint = torch.load('./transformer-translation-spoken.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    input_str = '나는 어제 음식을 먹었어.'
    # input_str = input('aa:')
    
    string = tokenizer.encode(input_str)
    string_len = len(string)
    pad_len = (max_length - string_len)
    encoder_input = torch.tensor(string + [tokenizer.pad_token_id]*pad_len)
    encoder_mask = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)
    
    target = torch.ones(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_input)
    
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