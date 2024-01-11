import matplotlib.pyplot as plt


def draw_plot(title, x, y, add_x_list=None, add_y_list=None, fig_size=None, 
              x_range=None, y_range=None, x_label=None, y_label=None, save_path=None):
    plt.title(title)
    
    if fig_size:
        plt.figure(figsize=fig_size)
    
    # 축 길이
    if x_range:
        x_min = x_range[0]
        x_max = x_range[1]
        plt.xlim(x_min, x_max)	# (최솟값, 최댓값)
    if y_range:
        y_min = y_range[0]
        y_max = y_range[1]
        plt.ylim(y_min, y_max)
    
    # 라벨링
    if x_label:
        plt.xlabel('xlabel')
    if y_label:
        plt.ylabel('ylabel')
    
    # plot
    plt.plot(x, y)
    
    if add_x_list and add_y_list:
        for add_x, add_y in zip(add_x_list, add_y_list):
            plt.plot(add_x, add_y)
    
    if save_path:
        plt.savefig(f'{save_path}')
        
    plt.clf()