from datetime import date, datetime, timedelta
import time


def time_measure(start):
    t = time.time() - start
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return h, m ,s


def get_date_list(schedule, year, mon_list, start_day_list, end_day_list):
    date_list = []
    if schedule:
        yesterday = date.today()# - timedelta(1)
        yesterday = str(yesterday)
        date_list = [yesterday]
    else:
        for mon in mon_list:
            start_day = start_day_list[mon-1]
            end_day = end_day_list[mon-1]
            for dd in range(start_day, end_day+1):
                dd = str(dd).zfill(2)
                mm = str(mon).zfill(2)
                date_list.append(f'{year}-{mm}-{dd}')
    return date_list