import shutil
import os
import sys
from tqdm import tqdm

def main():
    bus_name_list = os.listdir('/data/busmate/CNG')
    bus_name_list.sort()
    
    for bus_name in tqdm(bus_name_list):
        file_name_list = os.listdir(f'/data/busmate/CNG/{bus_name}')
        file_name_list.sort()
        
        os.makedirs(f'/data/busmate/data_raw/CNG/{bus_name}', exist_ok=True)
        for file_name in file_name_list:
            shutil.copy(
                f'/data/busmate/CNG/{bus_name}/{file_name}',
                f'/data/busmate/data_raw/CNG/{bus_name}/{file_name}'
            )

if __name__ == '__main__':
    main()