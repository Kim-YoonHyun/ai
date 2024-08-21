################## scheduler parameter ##################
year = 2024
mon_list = [6, 7]
#  month order = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
start_day_list = [22, 19, 24, 18,  1, 24,  2,  1,  1,  1,  1,  1]
end_day_list   = [22, 19, 30, 30, 31, 30, 16, 31, 30, 31, 30, 31]


################## basic paramerter ##################
root_path = '/home/kimyh/python/ai/time_series/generation/pytorch/VnAW/module_make_dataset/busmate'
root_data_path = '/data/busmate'
random_seed = 42

################## make dataset paramerter ##################
new_dataset_name = 'dataset_02_2'
column_json_name = 'column_json_8'
valid_json_name = 'valid_bus_name_17'
fuel_type = 'diesel_new'

data_interval = 1250

configuration = 'regression'
x_size = 5000
y_size = 5000

# configuration = 'bos time series'
# x_size = 5000
# y_size = 600

train_p = 90






















