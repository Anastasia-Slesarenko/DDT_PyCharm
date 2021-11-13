import numpy as np
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


def find_false_pixels(frames_0='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/'
                               'test_frames/cap_cut_crop.mp4/0000000000.jpg'):
    frame_with_name = Image.open(f'{frames_0}')
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

        # Записываем яркость искр
        for i in range(loc_idx.shape[0]):
            alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]
        # Соединяем координаты искр и их яркость на кадре в один массив
        # Сохраняем в txt файле с именем кадра
        all_ = np.append(loc_idx, np.array(alpha).reshape((len(alpha), 1)), axis=1)
        # x - all[:, 1]
        # y - all[:, 0]
        # alpha - all[:, 2]
        np.savetxt(fname=f'{path_frames}{int(frame[:-4])}.txt', X=all_, fmt='%d')

    return 1


def get_frames_with_discharges(path_frames='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/test_frames'
                                           '/cap_cut_crop.mp4/', num_brightpixels=10):
    os.chdir(path_frames)
    result = glob.glob('*.jpg')

    false_idx = find_false_pixels(frames_0='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm'
                                           '/test_frames/cap_cut_crop.mp4/0000000000.jpg')

    n_jobs = 20  # кол-во процессов
    pbar = tqdm(total=len(result))

    def update(out):
        pbar.update()

    pool = mp.Pool(n_jobs)  # открываем процессы
    need_func = partial(async_load_find,
                        path_frames=path_frames,
                        false_idx=false_idx,
                        num_brightpixels=num_brightpixels)

    for name_frame in sorted(result):
        pool.apply_async(need_func, args=(name_frame, ), callback=update)

    pool.close()
    pool.join()
    pbar.close()


if __name__ == '__main__':
    get_frames_with_discharges(
        path_frames='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/test_frames/cap_cut_crop.mp4/',
        num_brightpixels=10)