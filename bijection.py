import numpy as np
import os
from tqdm.auto import tqdm
import glob
import cv2


def new_coord_pix(path_file_pix, new_path_file_pix,  path_image0_p, path_image0_d):
    img_p = cv2.imread(path_image0_p)
    h_p, w_p, no_p = img_p.shape

    img_d = cv2.imread(path_image0_d)
    h_d, w_d, no_d = img_d.shape

    os.chdir(path_file_pix)
    result = glob.glob('*.txt')

    for name in tqdm(sorted(result)):
        coord_pixel = [(int(name[:-4]), np.loadtxt(name, dtype=int))]

        for t, cord in coord_pixel:
            try:
                y_old = np.array(cord)[:, 0]
                x_old = np.array(cord)[:, 1]
                x_new = []
                y_new = []
                for x, y in zip(x_old, y_old):

                    x_t = x/w_p
                    y_t = y/h_p
                    x_new += [int(x_t * w_d)]
                    y_new += [int(y_t * h_d)]
                new_cord = np.array([y_new, x_new]).T
                np.savetxt(fname=f'{new_path_file_pix}new{name}', X=new_cord, fmt='%d')

            except:
                y_old = np.array(cord)[0]
                x_old = np.array(cord)[1]
                x_t = int(x_old / w_p * w_d)
                y_t = int(y_old / h_p * h_d)
                new_cord = np.array([y_t, x_t]).T
                np.savetxt(fname=f'{new_path_file_pix}new{name}', X=new_cord, fmt='%d')


if __name__ == '__main__':
    f = 'PKD_02.02.22_part/1'
    try:
        new_coord_pix(path_file_pix=f'C:/Users/1/Desktop/VKR/Master/'
                                        f'data/silicone SPBU/{f}/test_frames/cap_cut_crop.mp4/',
                      new_path_file_pix='C:/Users/1/Desktop/VKR/Master/'
                                        f'data/silicone SPBU/{f}/test_frames_drops/cap_cut_crop_drops.mp4/',
                      path_image0_p='C:/Users/1/Desktop/VKR/Master/'
                                        f'data/silicone SPBU/{f}/test_frames/cap_cut_crop.mp4/0000000000.jpg',
                      path_image0_d='C:/Users/1/Desktop/VKR/Master/'
                                        f'data/silicone SPBU/{f}/test_frames_drops/cap_cut_crop_drops.mp4/0000000000.jpg')
    except Exception as e:
        print(e)
# def rotate(path_file_coord):
#
#     f = open(path_file_coord, mode='r', encoding='utf8', newline='\r\n')
#     coord_p = []
#     for i, line in enumerate(f):
#         coord_p += [list(map(int, line.split(',')))]
#     y_start = coord_p[0][1]
#     y_0 = coord_p[1][1]
#     x_start = coord_p[2][0]
#     y_end = coord_p[1][1]
#     x_end = coord_p[0][0]
#     x_0 = coord_p[1][0]
#
#     dx = x_end-x_0
#     dy = y_start-y_0
#     dl = np.sqrt((y_start-y_0)**2 + (x_end-x_0)**2)
#     sin_theta = dx/dl
#     cos_theta = dy/dl
#
#     h = y_end - y_start
#     w = x_end - x_start
#
#     return sin_theta, cos_theta, h, w
#
#
# def new_coord_pix(path_file_pix, new_path_file_pix,  path_file_coord_p, path_file_coord_d):
#
#     os.chdir(path_file_pix)
#     result = glob.glob('*.txt')
#
#     for name in tqdm(sorted(result)):
#         coord_pixel = [(int(name[:-4]), np.loadtxt(name, dtype=int))]
#
#         for t, cord in coord_pixel:
#             y_old = np.array(cord)[:, 0]
#             x_old = np.array(cord)[:, 1]
#
#             sin_theta_p, cos_theta_p, h_p, w_p = rotate(path_file_coord=path_file_coord_p)
#             x_new = []
#             y_new = []
#             for x, y in zip(x_old, y_old):
#
#                 # Левосторонняя система координат, поворот по часовой, масштабируем
#                 x_t = (x*cos_theta_p - y*sin_theta_p)/w_p
#                 y_t = (+ x*sin_theta_p + y*cos_theta_p)/h_p
#
#                 sin_theta_d, cos_theta_d, h_d, w_d = rotate(path_file_coord=path_file_coord_d)
#
#                 x_new += [x_t * w_d*cos_theta_d + y_t * h_d*sin_theta_d]
#                 y_new += [- x_t * w_d*sin_theta_d + y_t * h_d*cos_theta_d]
#
#             new_cord = np.array([y_new, x_new]).T
#             np.savetxt(fname=f'{new_path_file_pix}new{name}', X=new_cord, fmt='%d')
#
#
# if __name__ == '__main__':
#
#     new_coord_pix(path_file_pix='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/'
#                                 'DDT_PyCharm/test_frames/cap_cut_crop.mp4/',
#                   new_path_file_pix='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm'
#                                     '/DDT_PyCharm/test_frames_drops/cap_cut_crop_drops.mp4/',
#                   path_file_coord_p='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/coord_p.txt',
#                   path_file_coord_d='C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/coord_d.txt')
