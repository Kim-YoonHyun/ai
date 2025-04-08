import sys
import os
import pandas as pd
import numpy as np
np.random.seed(42)

def get_string(df):
    title_list = df['title'].to_list()
    contents_list = df['contents'].to_list()
    string_list = []
    for title, contents in zip(title_list, contents_list):
        string = f'{title} {contents}'
        
        while True:
            if '  ' in string:
                string = string.replace('  ', ' ')
            else:
                break
        string_list.append(string)
    return string_list


def get_sample(df, sample_num):
    train_idx_ary = np.random.randint(0, len(df), sample_num)
    train_sample_df = df.iloc[train_idx_ary, :]
    train_sample_df = train_sample_df.reset_index(drop=True)
    return train_sample_df


def main():
    # train_df = pd.read_csv('./data/ag_news_csv/train.csv', encoding='utf-8-sig')
    # train_string_list = get_string(train_df)
    # train_label_ary = train_df['class'].values - 1
    # new_train_df = pd.DataFrame([train_label_ary, train_string_list], index=['label', 'contents']).T
    # new_train_df.to_csv('./data/ag_news_csv/new_train.csv', index=False, encoding='utf-8-sig')

    
    # test_df = pd.read_csv('./data/ag_news_csv/test.csv', encoding='utf-8-sig')
    # test_string_list = get_string(test_df)
    # test_label_ary = test_df['class'].values - 1
    # new_test_df = pd.DataFrame([test_label_ary, test_string_list], index=['label', 'contents']).T
    # new_test_df.to_csv('./data/ag_news_csv/new_test.csv', index=False, encoding='utf-8-sig')
    
    
    train_sample_num = 3000
    test_sample_num = 300
    
    new_train_df = pd.read_csv('./data/ag_news_csv/new_train.csv', encoding='utf-8-sig')
    train_sample_df = get_sample(new_train_df, train_sample_num)
    train_sample_df.to_csv(f'./data/train_sample_{train_sample_num}.csv', index=False, encoding='utf-8-sig')
    
    new_test_df = pd.read_csv('./data/ag_news_csv/new_test.csv', encoding='utf-8-sig')
    test_sample_df = get_sample(new_test_df, test_sample_num)
    test_sample_df.to_csv(f'./data/test_sample_{test_sample_num}.csv', index=False, encoding='utf-8-sig')
    


if __name__ == '__main__':
    main()