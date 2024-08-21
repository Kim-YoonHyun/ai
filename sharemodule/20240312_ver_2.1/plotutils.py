import sys
import os
import matplotlib.pyplot as plt

COLOR_DICT = {
    'default':None,
    'blue':'b',
    'green':'b',
    'red':'r',
    'cyan':'c',
    'magenta':'m',
    'yellow':'y',
    'black':'k',
    'white':'w'
}

LINE_DICT = {
    'default':'-',
    'None':'None', 
    'line':'-',
    'dash':'--',
    'dot':':',
    'dash-dot':'-.'
}

MARKER_DICT = {
    'default':None,
    'None':None,
    'dot':'.',
    'pixel':',',
    'circle':'o',
    'triangle_down':'v',
    'triangle_up':'^',
    'triangle_left':'<',
    'triangle_right':'>',
    'tri_down':'1',
    'tri_up':'2',
    'tri_left':'3',
    'tri_right':'4',
    'square':'s',
    'pentagon':'p',
    'star':'*',
    'hexagon1':'h',
    'hexagon2':'H',
    'plus':'+',
    'x':'x',
    'diamond':'D',
    'thin_diamond':'d'
}

def get_style(line_style='default', line_size='default', line_color='default',
              marker_style='default', marker_size='default', marker_color='default',
              marker_border_size='default', marker_border_color='default'):
    
    # 라인 스타일
    try:
        ls = LINE_DICT[line_style]
    except KeyError:
        ls = '-'
    
    
    # 라인 크기
    if line_size == 'default':
        lw = None
    else:
        lw = line_size
    
    # 라인 색
    try:
        c = COLOR_DICT[line_color]
    except KeyError:
        c = None
    
    # 마커 형태
    try:
        marker = MARKER_DICT[marker_style]
    except KeyError:
        marker = None
    
    # 마커 크기
    if marker_size == 'default':
        ms = None
    else:
        ms = marker_size
        
    # 마커 색
    try:
        mfc = COLOR_DICT[marker_color]
    except KeyError:
        mfc = None
        
    # 마커 테두리 사이즈
    if marker_border_size == 'default':
        mew = None
    else:
        mew = marker_border_size
        
    # 마커 테두리 색
    try:
        mec = COLOR_DICT[marker_border_color]
    except KeyError:
        mec = None
    
    return ls, lw, c, marker, ms, mfc, mew, mec
    

def draw_plot(title, x, y, 
              line_style='default', line_size='default', line_color='default',
              marker_style='default', marker_size='default', marker_color='default',
              marker_border_size='default', marker_border_color='default',
              add_x_list=None, add_y_list=None, 
              fig_size=None, x_range=None, y_range=None, 
              label=None, save_path=None):
    
    plt.title(title)
    ls, lw, c, marker, ms, mfc, mew, mec = get_style(
        line_style=line_style, 
        line_size=line_size, 
        line_color=line_color,
        marker_style=marker_style, 
        marker_size=marker_size, 
        marker_color=marker_color,
        marker_border_size=marker_border_size, 
        marker_border_color=marker_border_color
    )
    
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
    
    
    
    
    # plot
    plt.plot(
        x, y, 
        ls=ls, 
        lw=lw, 
        c=c, 
        marker=marker, 
        ms=ms, 
        mfc=mfc, 
        mew=mew, 
        mec=mec,
        label=label
    )
    
    # 라벨링
    if label is not None:
        plt.xlabel('xlabel')
        plt.ylabel('ylabel')
        plt.legend()
    
    if add_x_list is not None and add_y_list is not None:
        for add_x, add_y in zip(add_x_list, add_y_list):
            plt.plot(add_x, add_y, ls=ls, marker=marker)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{title}.png')
        
    plt.clf()