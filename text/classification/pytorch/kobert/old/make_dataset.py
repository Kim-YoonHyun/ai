from mylocalmodules import dataloader as dutils
import os
import json

root_path = '/home/kimyh/python/project/complaints'
dataset_name = 'datasets_023'

dataset_info, train_data, val_data = dutils.make_json_dataset(
    load_path=f'{root_path}/datasets_raw/{dataset_name}.csv',
    train_p=0.8
)

os.makedirs(f'{root_path}/datasets/{dataset_name}', exist_ok=True)

with open(f'{root_path}/datasets/{dataset_name}/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent='\t', ensure_ascii=False)

with open(f'{root_path}/datasets/{dataset_name}/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent='\t', ensure_ascii=False)

with open(f'{root_path}/datasets/{dataset_name}/val_data.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, indent='\t', ensure_ascii=False)