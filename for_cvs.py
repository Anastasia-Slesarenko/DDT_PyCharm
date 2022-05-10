import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import re
from scipy.signal import savgol_filter



def sum_df_to_cvs(path_file_coord, name_cvs, name_cvs_reg, name_cvs_reg2):
    os.chdir(path_file_coord)
    result = glob.glob('*.txt')
    coord_pixel = []

    for name in tqdm(sorted(result)):
        coord_pixel += [(int(re.findall(r'\d+', name)[0]), np.loadtxt(name, dtype=int))]

    y_sum = []
    x_sum = []
    alpha_sum = []
    alpha_all = []
    t_sum = []
    t_sum_for = [] # для карты разрядов

    if name_cvs_reg != 'None':
        for t, cord in tqdm(coord_pixel):
            try:
                y = np.array(cord)[:, 0]
                x = np.array(cord)[:, 1]
                alpha_sum += list(np.array(cord)[:, 2])
                alpha_all += [sum(list(np.array(cord)[:, 2]))]
                y_sum += list(y)
                x_sum += list(x)
                t_sum += [t]
                t_sum_for += [t] * len(list(y))
            except:
                y = np.array(cord)[0]
                x = np.array(cord)[1]
                alpha_sum += [cord[2]]
                alpha_all += [cord[2]]
                y_sum += [y]
                x_sum += [x]
                t_sum += [t]
                t_sum_for += [t]

        t_sum_index = np.argsort(t_sum)
        t_sum = np.array(t_sum)[t_sum_index]
        alpha_all = np.array(alpha_all)[t_sum_index]

        not_reg = pd.DataFrame()
        not_reg['t_sum'] = t_sum
        not_reg['alpha_all'] = alpha_all
        not_reg.to_csv(name_cvs_reg2)
        step_t = 1
        t_reg = pd.DataFrame()
        t_reg['t_sum'] = np.arange(min(t_sum), max(t_sum), step_t)
        reg_data = t_reg.merge(not_reg, on='t_sum', how='left')
        threshold = 20 # уровень фона
        reg_data['alpha_all'] = reg_data['alpha_all'].fillna(threshold)
        reg_data.to_csv(name_cvs_reg)

    else:
        for t, cord in tqdm(coord_pixel):
            try:
                y = np.array(cord)[:, 0]
                x = np.array(cord)[:, 1]
                y_sum += list(y)
                x_sum += list(x)
                t_sum_for += [t] * len(list(y))  # ДЛЯ PLOTLY!!!!!
            except:
                y = np.array(cord)[0]
                x = np.array(cord)[1]
                y_sum += [y]
                x_sum += [x]
                t_sum_for += [t]

    sum_df = pd.DataFrame()
    sum_df['alpha_sum'] = alpha_sum
    sum_df['y_sum'] = y_sum
    sum_df['x_sum'] = x_sum
    sum_df['t_sum'] = t_sum_for
    sum_df.to_csv(name_cvs)


if __name__ == '__main__':
    f = ['PKD_02.02.22_part/1/', 'PKD_02.02.22_part/2/', 'PKD_09.02.22_part/4/', 'PKD_10.02.22_part/5/',
         'PKD_11.02.22_part/8/', 'PKD_18.02.22_part/10/']

    for n in f:
        sum_df_to_cvs(
            path_file_coord=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{n}test_frames/cap_cut_crop.mp4/',
            name_cvs='sum_df_new.cvs', name_cvs_reg='reg_data_new.cvs', name_cvs_reg2='not_reg_data_new.cvs')

    # sum_df_to_cvs(path_file_coord='C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_18.02.22_part/10'
    #                               '/test_frames_drops/cap_cut_crop_drops.mp4/',
    #               name_cvs='sum_df_fix_new.cvs', name_cvs_reg='None', name_cvs_reg2='None')
    # name_cvs = 'sum_df_fix.cvs', name_cvs_reg = 'None' для координат на кадрах с каплями из test_frames_drops
#
# if __name__ == '__main__':
#     path_file_coord = 'C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/2021-11-22 17-28-49/test_frames_drops/cap_cut_crop_drops.mp4/'
#     os.chdir(path_file_coord)
#     result = glob.glob('*.txt')
#     coord_pixel = []
#
#     for name in tqdm(sorted(result)):
#         coord_pixel += [(int(re.findall(r'\d+', name)[0]), np.loadtxt(name, dtype=int))]
#
#     y_sum = []
#     x_sum = []
#     t_sum_for = []
#     for t, cord in tqdm(coord_pixel):
#         try:
#             y = np.array(cord)[:, 0]
#             x = np.array(cord)[:, 1]
#             y_sum += list(y)
#             x_sum += list(x)
#             t_sum_for += [t] * len(list(y))  # ДЛЯ PLOTLY!!!!!
#         except:
#             y = np.array(cord)[0]
#             x = np.array(cord)[1]
#             y_sum += [y]
#             x_sum += [x]
#             t_sum_for += [t]
#     sum_df_fix = pd.DataFrame()
#     sum_df_fix['y_sum'] = y_sum
#     sum_df_fix['x_sum'] = x_sum
#     sum_df_fix['t_sum'] = t_sum_for
#     sum_df_fix.to_csv('sum_df_fix.cvs')