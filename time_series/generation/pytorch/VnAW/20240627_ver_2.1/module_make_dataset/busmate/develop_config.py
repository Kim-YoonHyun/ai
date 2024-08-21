################## scheduler parameter ##################
year = 2024
mon_list = [4, 5, 6]
#  month order = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
start_day_list = [22, 19, 24, 31,  1,  1,  1,  1,  1,  1,  1,  1]
end_day_list   = [22, 19, 30, 31, 31,  6, 31, 31, 30, 31, 30, 31]


################## basic paramerter ##################
# basic
root_path = '/home/kimyh/python/ai/time_series/generation/pytorch/VnAW/module_make_dataset/busmate'
root_data_path = '/data/busmate'
random_seed = 42

################## make dataset paramerter ##################
make_from_data_raw = True
new_dataset_name = 'dataset_02'
column_json_name = 'column_json_8'
except_json_name = 'except_bus_name_17'
# md_fuel_type = 'CNG'
fuel_type = 'diesel_new'
# md_fuel_type = 'CNG'

bos_len = 1200
# bos_num = 3
# y2x = False

data_interval = 2500
data_size = 5000
# x_seg_sec = 1200
# y_seg_sec = 1200
train_p = 90
# zero_speed_p = 50






















