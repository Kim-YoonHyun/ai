import sys
import os
import matplotlib.pyplot as plt

COLOR_DICT = {
    'default':None,
    'blue':'b',
    'green':'g',
    'red':'r',
    'cyan':'c',
    'magenta':'m',
    'yellow':'y',
    'black':'k',
    'white':'w',
    'orange':'#ffa500',
    'pink':'#ffc0cb',
    'khaki':'#f0e68c',
    'gold':'#ffd700',
    'skyblue':'#87ceeb',
    'navy':'#000080',
    'lightgreen':'#90ee90',
    'olive':'#808000',
    'violet':'#ee82ee',
    'gray':'#808080',
    'brown':'#a52a2a'
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
    

def draw_plot(title, x, y, title_font_size=13, x_font_size=13, y_font_size=13, 
              line_style='default', line_size='default', line_color='default',
              marker_style='default', marker_size='default', marker_color='default',
              marker_border_size='default', marker_border_color='default',
              add_x_list=None, add_y_list=None, add_color_list=None,
              fig_size=None, x_range=None, y_range=None, 
              focus_start_list=None, focus_end_list=None, focus_color_list=None, alpha_list=None,
              label=None, save_path=None):
    
    
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
    
    plt.title(title, fontdict={'fontsize':title_font_size})
    
    # x, y 축 글자 크기
    plt.xticks(fontsize=x_font_size)
    plt.yticks(fontsize=y_font_size)
    
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
    
    # 추가 
    
    if add_x_list is not None and add_y_list is not None:
        
        if add_color_list is None:
            for add_x, add_y in zip(add_x_list, add_y_list):
                plt.plot(
                    add_x, add_y, 
                    ls=ls, 
                    marker=marker
                )
        else:
            for add_x, add_y, add_color in zip(add_x_list, add_y_list, add_color_list):
                _, _, add_c, _, _, _, _, _ = get_style(line_color=add_color)
                plt.plot(
                    add_x, add_y, 
                    ls=ls, 
                    marker=marker,
                    c=add_c
                )
                
    # 포커싱
    if focus_start_list is not None and focus_end_list is not None:
        if focus_color_list is None:
            focus_color_list = ['gray'] * len(focus_start_list)
        if alpha_list is None:
            alpha_list = [0.2] * len(focus_start_list)
        for focus_start, focus_end, focus_c, alpha in zip(focus_start_list, focus_end_list, focus_color_list, alpha_list):
            plt.axvspan(focus_start, focus_end, facecolor=focus_c, alpha=alpha)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{title}.png')
    
    plt.close('all') 
    
    
def draw_subplot(image_title, sub_row_idx, sub_col_idx, 
                 title_list, x_list, y_list, 
                 title_font_size=13, x_font_size=13, y_font_size=13,
                 x_range_list=None, y_range_list=None, 
            #   line_style='default', line_size='default', line_color='default',
            #   marker_style='default', marker_size='default', marker_color='default',
            #   marker_border_size='default', marker_border_color='default',
            #   add_x_list=None, add_y_list=None, add_color_list=None,
              fig_size=None, # x_range=None, y_range=None, 
              focus_start_list=None, focus_end_list=None, focus_color_list=None, alpha_list=None,
              label=None, save_path=None):
    
    # ls, lw, c, marker, ms, mfc, mew, mec = get_style(
    #     line_style=line_style, 
    #     line_size=line_size, 
    #     line_color=line_color,
    #     marker_style=marker_style, 
    #     marker_size=marker_size, 
    #     marker_color=marker_color,
    #     marker_border_size=marker_border_size, 
    #     marker_border_color=marker_border_color
    # )
    
    if fig_size is not None:
        fig, axs = plt.subplots(sub_row_idx, sub_col_idx, figsize=fig_size)
    else:
        fig, axs = plt.subplots(sub_row_idx, sub_col_idx)
    
    for i, ax in enumerate(axs.flat):
        x_ = x_list[i]
        y_ = y_list[i]
        ax.plot(x_, y_)
        ax.set_title(title_list[i], fontsize=title_font_size)
        
        if x_range_list is not None:
            ax.set_xlim(x_range_list[i], fontsize=x_font_size)
            
        if y_range_list is not None:
            ax.set_ylim(y_range_list[i])#, fontsize=y_font_size)
        ax.tick_params(axis='x', labelsize=x_font_size)
        ax.tick_params(axis='y', labelsize=y_font_size)
        
        # 포커싱
        if focus_start_list is not None and focus_end_list is not None:
            if focus_color_list is None:
                focus_color_list = ['gray'] * len(focus_start_list)
            if alpha_list is None:
                alpha_list = [0.2] * len(focus_start_list)
            for focus_start, focus_end, focus_c, alpha in zip(focus_start_list, focus_end_list, focus_color_list, alpha_list):
                ax.axvspan(focus_start, focus_end, facecolor=focus_c, alpha=alpha)
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.4, hspace=0.7, top=0.2, bottom=0.1)  # wspace: 수평 간격, hspace: 수직 간격
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{image_title}.png')
    
    plt.close(fig)    
    # plt.clf()
    
    
    
    
