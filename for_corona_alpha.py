import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
hold = 15
hold_min = 10
hold_max = 15

def count_bright_pixels(frame, crit_pix=hold):
    count = np.sum(frame >= crit_pix)
    return count


# def find_false_pixels(frames_0):
#     frame_with_name = Image.open(f'{frames_0}')
#     #width, height = frame_with_name.size
#     # loc_x_start = width / 4
#     # loc_x_end = 3 * width / 4
#     frame_gray = ImageOps.grayscale(frame_with_name)
#     # Убираем шум через медианный фильтр
#     frame_gray_from_filter = frame_gray.filter(ImageFilter.MedianFilter(3))
#     frame_gray_np = np.array(frame_gray_from_filter)
#     false_idx = np.stack(np.where(frame_gray_np >= hold), axis=1)
#
#     return false_idx


# def row_diff(loc_idx_old, fals):
#     nrows, ncols = loc_idx_old.shape
#     dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
#              'formats': ncols * [loc_idx_old.dtype]}
#     _, delete_a_rows, _ = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype), return_indices=True)
#
#     # Координаты пискелей искр без битых
#     loc_idx = np.delete(loc_idx_old, delete_a_rows, axis=0)
#
#     return loc_idx


def async_load_find(frame, path_frames, num_brightpixels=10):
    frame_with_name = Image.open(f'{path_frames}{frame}')
    frame_gray = ImageOps.grayscale(frame_with_name)
    # Убираем шум через медианный фильтр
    #frame_gray_from_filter = frame_gray.filter(ImageFilter.MedianFilter(3))
    frame_gray_np = np.array(frame_gray)

    if count_bright_pixels(frame_gray_np, crit_pix=hold) > num_brightpixels:
        alpha = []
        loc_idx = np.stack(np.where(frame_gray_np > hold), axis=1)

        # Убираем битые пиксели на изображении
        if loc_idx.shape[0] >= 5:

            # Записываем яркость искр
            for i in range(loc_idx.shape[0]):
                alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]

    return np.sum(alpha), len(alpha)


def get_frames_with_discharges(path_frames, path_frames_0, num_brightpixels=10):
    os.chdir(path_frames)
    result = glob.glob('*.jpg')

    total_alpha = []
    len_alpha = []

    for name_frame in sorted(result):
        total_alpha += [async_load_find(frame=name_frame, path_frames=path_frames, num_brightpixels=10)[0]]
        len_alpha += [async_load_find(frame=name_frame, path_frames=path_frames, num_brightpixels=10)[1]]

    return np.mean(total_alpha), np.mean(len_alpha)


if __name__ == '__main__':
    dat_fil = 'PKD_02.02.22_part'
    nv = [2, 3, 4, 5, 6]
    t_alpha = []

    for name_video in tqdm(nv):
        t_alpha += [get_frames_with_discharges(
            path_frames=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                        f'/test_frames_{name_video}/cap_cut_{name_video}.mp4/',
            path_frames_0=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                          '/test_frames_1/cap_cut_1.mp4/0000000000.jpg',
            num_brightpixels=10
        )[0]]

    alpha_corona = pd.DataFrame()
    alpha_corona['alpha'] = t_alpha
    alpha_corona.to_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD/alpha_corona_15.csv',
                        index=False)

    number_pix = []
    for name_video in tqdm(nv):
        number_pix += [get_frames_with_discharges(
            path_frames=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                        f'/test_frames_{name_video}/cap_cut_{name_video}.mp4/',
            path_frames_0=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD'
                          '/test_frames_1/cap_cut_1.mp4/0000000000.jpg',
            num_brightpixels=10
        )[1]]

    n = np.mean(number_pix)
    print(n)
    # np.savetxt(fname=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{dat_fil}/CD/number_corona_15.txt', X=n)
