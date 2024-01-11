import sys
import os
import pandas as pd
import numpy as np
import random
# random.seed(42)


def main():
    dataset_path = '/home/kimyh/python/project/bscr/dataset/ECU_BatteryVolt'
    new_dataset_path = '/home/kimyh/python/project/bscr/dataset/dataset_test'
    train_val = ['train', 'val']

    for tv in train_val:
        cond_data_path = f'{dataset_path}/{tv}/condition'
        target_data_path = f'{dataset_path}/{tv}/target'
        
        x_data_path = f'{new_dataset_path}/{tv}/x'
        mark_data_path = f'{new_dataset_path}/{tv}/mark'
        os.makedirs(x_data_path, exist_ok=True)
        os.makedirs(mark_data_path, exist_ok=True)
        
        
        data_name_list = os.listdir(f'{dataset_path}/{tv}/target')
        data_name_list.sort()
        data_name_list = random.sample(data_name_list, 5)
    
        
        for data_name in data_name_list:
            cond_df = pd.read_csv(f'{cond_data_path}/{data_name}', encoding='utf-8-sig') 
            target_df = pd.read_csv(f'{target_data_path}/{data_name}', encoding='utf-8-sig') 
            cond_df = cond_df.iloc[:12, 3:7]
            target_df = target_df.iloc[:12, :]
            
            rand_num = random.randint(0, 100)
            rand_num = rand_num % 5
            if rand_num == 0:
                rand_num = 1
            cond_df.iloc[-rand_num:] = -10000
            target_df.iloc[-rand_num:] = -10000
            
            cond_df.to_csv(f'{mark_data_path}/{data_name}', index=False, encoding='utf-8-sig')
            target_df.to_csv(f'{x_data_path}/{data_name}', index=False, encoding='utf-8-sig')
            # print(df)


def main2():
    dataset_path = '/home/kimyh/python/project/bscr/dataset/dataset_test'
    train_val = ['train', 'val']
    x_mark = ['x', 'mark']
    for tv in train_val:
        for xm in x_mark:
            data_name_path = f'{dataset_path}/{tv}/{xm}'
            data_name_list = os.listdir(data_name_path)
            data_name_list.sort()
            for data_name in data_name_list:
                df = pd.read_csv(f'{data_name_path}/{data_name}', encoding='utf-8-sig')
                ary = df.to_numpy()
                ary = np.round(ary, 3)
                ary = np.where(ary == -10000, -100, ary)
                df = pd.DataFrame(ary)
                df.to_csv(f'{data_name_path}/{data_name}', index=False, encoding='utf-8-sig')
                
                
def main3():
    import torch
    import numpy as np
    i = 2
    j = 3
    k = 2
    l = 2
    a = torch.tensor(np.ones((i, j, k, l)))
    
    m = 2
    n = 3
    o = 2 
    p = 2    
    b = torch.tensor(np.ones((m, n, o, p)))
    # print(a)
    # print(b)
    # c = torch.einsum("blhe,bshe->bhls"
    c =   torch.einsum('ijkl,mnop->ikjn', a, b)
    # c = torch.einsum("ijkl,inkp->ikjn"
    print(a.size())
    print(b.size())
    print(c.size())
    print(c)
if __name__ == '__main__':
    # main()
    # main2()
    main3()