import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os
import glob
from tqdm import tqdm
import cv2
import imutils
import matplotlib.pyplot as plt


def count_bright_pixels(frame, crit_pix=20):

    # shape[0] - количество сторк
    # shape[1] - количество столбцов
    count = 0
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] > crit_pix:
                count += 1
    return count


if __name__ == '__main__':
    sum_df = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_18.02.22_part/10/test_frames'
                         '/cap_cut_crop.mp4/sum_df.cvs', index_col=0)
    # frame_with_name0 = Image.open('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/1'
    #                              '/test_frames/cap_cut_crop.mp4/0000000000.jpg')
    # frame_with_name = Image.open('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/1'
    #                              '/test_frames/cap_cut_crop.mp4/0000065000.jpg')
    # frame_gray = ImageOps.grayscale(frame_with_name)
    # frame_gray = frame_gray.filter(ImageFilter.MedianFilter(3))
    # frame_gray_np = np.array(frame_gray)
    #
    # frame_gray0 = ImageOps.grayscale(frame_with_name0)
    # frame_gray0 = frame_gray0.filter(ImageFilter.MedianFilter(3))
    # frame_gray_np0 = np.array(frame_gray0)
    t_criter = sum_df['t_sum'].max()
    sum_df_n = sum_df
    sum_max = sum_df[sum_df['alpha_sum'] == sum_df['alpha_sum'].max()]
    t_light = sum_max[sum_max['t_sum'] <= t_criter]['t_sum'].unique()
    print(len(t_light))
    l = []
    for t in t_light:
        l += [len(sum_max[sum_max['t_sum'] == t]['y_sum'].values)]

    # Выводим количество пикселей с яркостью 255 на каждом кадре, какие значения есть
    # В первом опыте есть 23, находим на каком это фрейме было и рисуем эту искру
    print(l)
    ind = np.where(np.array(l) == 3)[0][0]
    # print(ind)
    #print(sum_max[sum_max['t_sum'] == t_light[ind]])
    sum_df_n = sum_df[sum_df['t_sum'] == t_light[ind]]
    sum_df_n = sum_df_n[sum_df_n['alpha_sum'] >= 250]
    fig1, ax1 = plt.subplots()
    lc = ax1.scatter(x=sum_df_n['x_sum'], y=sum_df_n['y_sum'], c=sum_df_n['alpha_sum'])
    lc.set_array(sum_df_n['alpha_sum'])
    lin = ax1.add_collection(lc)
    fig1.colorbar(lin, ax=ax1).ax.tick_params(labelsize=15)
    plt.show()

    al_alpha = sum_df[sum_df['t_sum'] <= t_criter]['alpha_sum'].sum()
    hold_alpha = np.sum(l)*255
    p = hold_alpha/al_alpha*100
    print(hold_alpha, al_alpha, p)

    # width, height = frame_with_name.size
    # loc_x_start = width / 4
    # loc_x_end = 3 * width / 4
    # alpha = []
    #
    # false_idx = np.stack(np.where(frame_gray_np0 >= 10), axis=1)
    #
    # if count_bright_pixels(frame_gray_np, crit_pix=10) > 10:
    #     loc_idx_old = np.stack(np.where(frame_gray_np > 20), axis=1)
    #
    #     nrows, ncols = loc_idx_old.shape
    #     dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
    #              'formats': ncols * [loc_idx_old.dtype]}
    #     loc_idx = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype))
    #     _, delete_a_rows, _ = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype), return_indices=True)
    #
    #     # Пискели без битых
    #     loc_idx = np.delete(loc_idx_old, delete_a_rows, axis=0)
    #
    #     to_del = []
    #     for k in range(loc_idx.shape[0]):
    #         if (loc_idx[k][1] < int(loc_x_start)) | (loc_idx[k][1] > int(loc_x_end)):
    #             to_del += [k]
    #             print(k)
    #     loc_idx = np.delete(loc_idx, to_del, axis=0)
    #
    #     for i in range(loc_idx.shape[0]):
    #         alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]
    #     all_ = np.append(loc_idx, np.array(alpha).reshape((len(alpha), 1)), axis=1)
    # _, ax = plt.subplots()
    # imgplot = ax.imshow(frame_gray_np, cmap=plt.get_cmap('gray'))
    # in_df = pd.DataFrame()
    # in_df['x'] = all_[:, 1]
    # in_df['y'] = all_[:, 0]
    # in_df['alpha'] = all_[:, 2]
    # new = in_df[in_df['y'] <= 30]
    #ax.scatter(in_df['x'], in_df['y'], fc='r', marker='.', s=1)
    # ax.scatter(new['x'], new['y'], fc='r', marker='.', s=1)
    # ax.bar(new['y'], new['alpha'])
    # alpha_max = new['alpha'].max()
    # ax.axhline(255, c='r', ls='-.', lw=1, label=f'$alpha ={int(alpha_max)}$')
    # ax.legend()
    # plt.show()
    #print(in_df[in_df['alpha'] == in_df['alpha'].max()])

