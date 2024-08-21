################## scheduler parameter ##################
year = 2024
mon_list = [6, 7]
#  month order = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
start_day_list = [22, 19, 24, 18,  1,  7,  1,  1,  1,  1,  1,  1]
end_day_list   = [22, 19, 30, 30, 31, 30, 16, 31, 30, 31, 30, 31]


################## basic paramerter ##################
# basic
root_path = '/home/kimyh/python/ai/time_series/generation/pytorch/VnAW/module_make_dataset/busmate'
root_data_path = '/data/busmate'
random_seed = 42

################## make dataset paramerter ##################
make_from_data_raw = True
new_dataset_name = 'dataset_02_1'
column_json_name = 'column_json_8'
except_json_name = 'except_bus_name_17'
# md_fuel_type = 'CNG'
fuel_type = 'diesel_new'
# md_fuel_type = 'CNG'

# bos_len = 1200
# bos_num = 3
# eos_num = 4
# y2x = False

data_interval = 1250

configuration = 'regression'
x_size = 5000
y_size = 5000
# configuration = 'bos time series'
# x_size = 5000
# y_size = 600

# x_seg_sec = 1200
# y_seg_sec = 1200
train_p = 90
# zero_speed_p = 50






















