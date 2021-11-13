import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from tqdm.auto import tqdm


path_file_coord = 'C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/test_frames/cap_cut_crop.mp4/'
os.chdir(path_file_coord)
result = glob.glob('*.txt')
coord_pixel = []
y = []
t = []

for name in tqdm(sorted(result)):
    coord_pixel += [(int(name[:-4]), np.loadtxt(name, dtype=int))]

y_sum = []
x_sum = []
alpha_sum = []
t_sum = []
alpha_all = []

fig, ax = plt.subplots(nrows=1,ncols=3)
for t, cord in tqdm(coord_pixel):
    y = np.array(cord)[:, 0]
    x = np.array(cord)[:, 1]
    alpha_sum += list(np.array(cord)[:, 2])
    alpha_all += [sum(list(np.array(cord)[:, 2]))]
    y_sum += list(y)
    x_sum += list(x)
    t_sum += [t]
    # t_sum_for += [t] * len(list(y))  # ДЛЯ PLOTLY!!!!!
    ax[0].scatter(t*np.ones(y.shape[0]), y, s=8, c='k', marker=',')
ax[0].invert_yaxis()
ax[0].grid()
ax[0].set_title('Зависимость разрядов вдоль y-координаты от времени')
ax[0].set_xlabel("Номер кадра")
ax[0].set_ylabel("y-координата вдоль поверхности образца")

y_uniq, number_y = np.unique(y_sum, return_index=False, return_inverse=False, return_counts=True, axis=None)
ax[1].plot(number_y, y_uniq, '-', c='r')
ax[1].invert_yaxis()
ax[1].grid()
ax[1].set_title('Распределение яркости разрядов по длине образца')
ax[1].set_xlabel("Количество пикселе")
ax[1].set_ylabel("y-координата вдоль поверхности образца")

t_sum_index = np.argsort(t_sum)
t_sum = np.array(t_sum)[t_sum_index]
alpha_all = np.array(alpha_all)[t_sum_index]
ax[2].plot(t_sum, alpha_all, '-', c='r')
ax[2].grid()
ax[2].set_title('Зависимость яркости разрядов от времени')
ax[2].set_xlabel("Номер кадра")
ax[2].set_ylabel("Суммарная яркость на кадре")
plt.subplots_adjust(wspace=0.4)
plt.show()
#========================================================================
# Часть кода записи данных в cvs для построения карты разрядов
# sum_df = pd.DataFrame()
# sum_df['alpha_sum'] = alpha_sum
# sum_df['y_sum'] = y_sum
# sum_df['x_sum'] = x_sum
# sum_df['t_sum'] = t_sum_for
# sum_df.to_csv('sum_df.cvs')


