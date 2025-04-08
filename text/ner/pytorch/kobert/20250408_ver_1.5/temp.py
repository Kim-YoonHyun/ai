import sys
import os
import json


def main():
    with open('./kobert_token2idx_dict.json', 'r', encoding='utf-8-sig') as f:
        kobert_tokenizer_dict = json.load(f)

    
    tokenizer_dict = {}
    n = 0
    for key, value in kobert_tokenizer_dict.items():
        
        if key in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']:
            tokenizer_dict[key] = n
            n += 1
            
        elif '▁' not in key:
            tokenizer_dict[key] = n
            n += 1
            
    print(tokenizer_dict)
    with open('./tokenizer_dict.json', 'w', encoding='utf-8-sig') as f:
        json.dump(tokenizer_dict, f, indent='\t', ensure_ascii=False)

if __name__ == '__main__':
    main()