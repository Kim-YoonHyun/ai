################## scheduler parameter ##################
year = 2024
mon_list = [3, 4]
#  month order = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
start_day_list = [22, 19, 22,  1,  2,  1,  1,  1,  1,  1,  1,  1]
end_day_list   = [22, 19, 31, 17,  4, 30, 31, 31, 30, 31, 30, 31]


################## basic paramerter ##################
# basic
root_path = '/home/kimyh/python/project/busmate'
root_save_path = '/data/busmate'
random_seed = 42

################## make dataset paramerter ##################
make_from_data_raw = True
new_dataset_name = 'dataset_01'
column_json = 'column_json_7'
except_json = 'except_bus_name_8'
# md_fuel_type = 'CNG'
md_fuel_type = 'diesel_new'
# md_fuel_type = 'CNG'

# bos_num = 3
y2x = False

interval_sec = 360
x_seg_sec = 1200
y_seg_sec = 1200
train_p = 90
zero_speed_p = 50






















