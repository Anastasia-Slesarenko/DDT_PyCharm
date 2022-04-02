import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def count_bright_pixels(frame, crit_pix=20):

    # shape[0] - количество сторк
    # shape[1] - количество столбцов
    count = 0
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] > crit_pix:
                count += 1
    return count


def find_false_pixels(frames_0):
    frame_with_name = Image.open(f'{frames_0}')
    width, height = frame_with_name.size
    # loc_x_start = width / 4
    # loc_x_end = 3 * width / 4
    frame_gray = ImageOps.grayscale(frame_with_name)
    # Убираем шум через медианный фильтр
    frame_gray_from_filter = frame_gray.filter(ImageFilter.MedianFilter(3))
    frame_gray_np = np.array(frame_gray_from_filter)
    false_idx = np.stack(np.where(frame_gray_np >= 10), axis=1)

    return false_idx


def row_diff(loc_idx_old, false_idx):
    nrows, ncols = loc_idx_old.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [loc_idx_old.dtype]}
    _, delete_a_rows, _ = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype), return_indices=True)

    # Координаты пискелей искр без битых
    loc_idx = np.delete(loc_idx_old, delete_a_rows, axis=0)

    return loc_idx


def async_load_find(frame, path_frames, false_idx, num_brightpixels=10):
    frame_with_name = Image.open(f'{path_frames}{frame}')
    frame_gray = ImageOps.grayscale(frame_with_name)
    # Убираем шум через медианный фильтр
    frame_gray_from_filter = frame_gray.filter(ImageFilter.MedianFilter(3))
    frame_gray_np = np.array(frame_gray_from_filter)

    if count_bright_pixels(frame_gray_np, crit_pix=20) > num_brightpixels:
        alpha = []
        loc_idx = np.stack(np.where(frame_gray_np > 20), axis=1)

        # Убираем битые пиксели на изображении
        loc_idx = row_diff(loc_idx, false_idx)
        if loc_idx.shape[0] != 0:

            # Записываем яркость искр
            for i in range(loc_idx.shape[0]):
                alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]

    return np.sum(alpha)


def get_frames_with_discharges(path_frames, path_frames_0, num_brightpixels=10):
    os.chdir(path_frames)
    result = glob.glob('*.jpg')

    false_idx = find_false_pixels(frames_0=path_frames_0)
    total_alpha = []

    for name_frame in sorted(result):
        total_alpha += [async_load_find(frame=name_frame, path_frames=path_frames,
                                        false_idx=false_idx, num_brightpixels=10)]

    return np.mean(total_alpha)


if __name__ == '__main__':
    dat_fil = 'PKD_18.02.22_part'
    nv = [2, 3, 4, 5, 6]
    t_alpha = []

    for name_video in tqdm(nv):
        t_alpha += [get_frames_with_discharges(
            path_frames=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                        f'/test_frames_{name_video}/cap_cut_{name_video}.mp4/',
            path_frames_0=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                          '/test_frames_1/cap_cut_1.mp4/0000000000.jpg',
            num_brightpixels=10
        )]

    alpha_corona = pd.DataFrame()
    alpha_corona['alpha'] = t_alpha
    alpha_corona.to_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD/alpha_corona.csv',
                        index=False)