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
    hold = 20
    path_frames = 'C:/Users/1/Desktop/VKR/Master/data/new silicone/test_frames_1/cap_cut_1.mp4/'
    os.chdir(path_frames)
    result = glob.glob('*.jpg')
    index = []
    x = []
    y = []
    alpha =[]
    number_frames = []
    # Установить номер интересующего фрейма для отрисовки найденных пикселей в листе new
    new = [sorted(result)[20], sorted(result)[707]]
    name_frame = sorted(result)[20]
    frame_with_name = Image.open(f'{path_frames}{name_frame}')
    frame_gray = ImageOps.grayscale(frame_with_name)
    #frame_gray = frame_gray.filter(ImageFilter.MedianFilter(3))
    frame_gray_np = np.array(frame_gray)
    width, height = frame_with_name.size
    alpha = []
    loc_idx = np.stack(np.where(frame_gray_np >= hold), axis=1)
    for i in range(loc_idx.shape[0]):
        alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]
    all_ = np.append(loc_idx, np.array(alpha).reshape((len(alpha), 1)), axis=1)

    _, ax = plt.subplots()
    # Указать путь у интересующему фрейму
    frame_with_name = Image.open('C:/Users/1/Desktop/VKR/Master/data/new silicone/test_frames_1/cap_cut_1.mp4/00000020020.jpg')
    frame_gray = ImageOps.grayscale(frame_with_name)
    frame_gray_np = np.array(frame_gray)
    imgplot = ax.imshow(frame_gray_np, cmap=plt.get_cmap('gray'))
    ax.scatter(all_[:, 1], all_[:, 0], fc='r', marker='.', s=1)
    plt.show()
    print(len(all_[:, 1]))

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os
import glob
from tqdm import tqdm
import cv2
import imutils
import matplotlib.pyplot as plt


#
# def count_bright_pixels(frame, crit_pix=20):
#
#     # shape[0] - количество сторк
#     # shape[1] - количество столбцов
#     count = 0
#     for i in range(frame.shape[0]):
#         for j in range(frame.shape[1]):
#             if frame[i][j] > crit_pix:
#                 count += 1
#     return count

# if __name__ == '__main__':
#     hold = 15
#     num = 10
#     # path_frames = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/CD/test_frames_2/cap_cut_2.mp4/'
#     # os.chdir(path_frames)
#     # result = glob.glob('*.jpg')
#     index = []
#     x = []
#     y = []
#     alpha =[]
#     number_frames = []
#     f0 = 'C:/Users/1/Desktop/VKR/Master/data/new silicone/test_frames_1/cap_cut_1.mp4/'
#     f = 'C:/Users/1/Desktop/VKR/Master/data/new silicone/test_frames_1/cap_cut_1.mp4/'
#     path_frames = [f0, f]
#     # Установить номер интересующего фрейма для отрисовки найденных пикселей в листе new
#     new = ['0000000000.jpg', '0000000001.jpg', '0000485811.jpg']
#     for name_frame, path_frames in zip(new, path_frames): #tqdm(list(sorted(result))[0]):
#         frame_with_name = Image.open(f'{path_frames}{name_frame}')
#         frame_gray = ImageOps.grayscale(frame_with_name)
#         #frame_gray = frame_gray.filter(ImageFilter.MedianFilter(3))
#         frame_gray_np = np.array(frame_gray)
#         width, height = frame_with_name.size
#         loc_x_start = width/4
#         # loc_x_end = 3*width/4
#         loc_x_end = width / 2
#         alpha = []
#
#         if int(name_frame[:-4]) == 0:
#             false_idx = np.stack(np.where(frame_gray_np >= hold), axis=1)
#         if count_bright_pixels(frame_gray_np, crit_pix=hold) > num:
#             loc_idx_old = np.stack(np.where(frame_gray_np > hold), axis=1)
#
#             nrows, ncols = loc_idx_old.shape
#             dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
#                      'formats': ncols * [loc_idx_old.dtype]}
#             loc_idx = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype))
#             _, delete_a_rows, _ = np.intersect1d(loc_idx_old.view(dtype), false_idx.view(dtype), return_indices=True)
#
#             # Пискели без битых
#             loc_idx = np.delete(loc_idx_old, delete_a_rows, axis=0)
#
#             to_del = []
#             for k in range(loc_idx.shape[0]):
#                 # if (loc_idx[k][1] < int(loc_x_start)) | (loc_idx[k][1] > int(loc_x_end)):
#                 if loc_idx[k][1] > int(loc_x_end):
#                     to_del += [k]
#                     print(k)
#             loc_idx = np.delete(loc_idx, to_del, axis=0)
#
#             for i in range(loc_idx.shape[0]):
#                 alpha += [frame_gray_np[loc_idx[i][0], loc_idx[i][1]]]
#             all_ = np.append(loc_idx, np.array(alpha).reshape((len(alpha), 1)), axis=1)
#
#     _, ax = plt.subplots()
#     # Указать путь у интересующему фрейму
#     frame_with_name = Image.open('C:/Users/1/Desktop/VKR/Master/data/new silicone/test_frames_1/cap_cut_1.mp4/0000485811.jpg')
#     frame_gray = ImageOps.grayscale(frame_with_name)
#     #frame_gray = frame_gray.filter(ImageFilter.MedianFilter(3))
#     frame_gray_np = np.array(frame_gray)
#     imgplot = ax.imshow(frame_gray_np, cmap=plt.get_cmap('gray'))
#     ax.scatter(all_[:, 1], all_[:, 0], fc='r', marker='.', s=1)
#     plt.show()
#     print(len(all_[:, 1]))
