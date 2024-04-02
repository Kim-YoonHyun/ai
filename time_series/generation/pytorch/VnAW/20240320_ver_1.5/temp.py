import sys
import numpy as np
import pandas as pd
a = [1, 1, np.nan, np.nan, 0, 0, np.nan, np.nan, 1, 1, np.nan, np.nan, 1, 1, np.nan, np.nan, 0, 0, np.nan, np.nan, 0, 0]

new_list = []
for i in a:
    if i == 1:
        new_list.append('남해고속도로')
    elif i == 0:
        new_list.append('대전통영고속도로')
    else:
        new_list.append(np.nan)

df = pd.DataFrame(new_list, columns=['polygon_name'])

def do_fill(df):
    temp_df = df.iloc[:, -1:]
    temp_df_up = temp_df.fillna(method='ffill')
    temp_df_down = temp_df.fillna(method='bfill')
    up = np.squeeze(temp_df_up.values)
    down = np.squeeze(temp_df_down.values)
    
    up = np.where(up == '남해고속도로', 1, 0)
    down = np.where(down == '남해고속도로', 1, 0)
    
    total = up + down
    
    total = np.where(total == 2, '남해고속도로', total)
    total = np.where(total == '0', '대전통영고속도로', total)
    total = np.where(total == '1', np.nan, total)
    
    df['polygon_name'] = total
    return df

# print(up + down)
print(df)
new_df = do_fill(df)
print('-------------------------')
print(new_df)