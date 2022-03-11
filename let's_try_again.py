import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def integ(I, t):
    I_s = []
    for t_i in t:
        ind = int(np.where(t == t_i)[0])
        I_s += [np.trapz(I[:ind], dx=1)]
    return I_s


if __name__ == '__main__':

    reg_data = pd.read_csv('C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/PKD_18.02.22_part/10/test_frames'
                           '/cap_cut_crop.mp4/reg_data.cvs', index_col=0)

    sum_df = pd.read_csv('C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/PKD_18.02.22_part/10/test_frames'
                         '/cap_cut_crop.mp4/sum_df.cvs', index_col=0)

    fig1, ax1 = plt.subplots(figsize=(13, 8))
    ax1.scatter(sum_df['t_sum'], sum_df['y_sum'], s=8, c='k', marker=',')
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_title('Зависимость разрядов вдоль y-координаты от времени', fontsize=15)
    ax1.set_xlabel("Номер кадра", fontsize=15)
    ax1.set_ylabel("y-координата вдоль поверхности образца", fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)
    plt.show()

    sum_df = sum_df.groupby('y_sum', as_index=False)['alpha_sum'].sum()

    fig2, ax2 = plt.subplots(figsize=(13, 8))
    ax2.plot(sum_df['alpha_sum'], sum_df['y_sum'], '-', c='r')
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('Распределение яркости разрядов по длине образца', fontsize=15)
    ax2.set_xlabel("Яркость", fontsize=15)
    ax2.set_ylabel("y-координата вдоль поверхности образца", fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(13, 8))
    alpha_all_reg = np.array(reg_data['alpha_all'])
    t_sum_reg = np.array(reg_data['t_sum'])
    alpha_all_savg = savgol_filter(alpha_all_reg, 2001, 1)
    ax3.semilogy(t_sum_reg, alpha_all_reg, '-', c='grey', label='True')
    ax3.semilogy(t_sum_reg, alpha_all_savg, '-', c='b', label='savgol_filter, window_size=2001')
    ax3.grid()
    ax3.set_title('Зависимость яркости разрядов от времени', fontsize=15)
    ax3.set_xlabel("Номер кадра", fontsize=15)
    ax3.set_ylabel("Суммарная яркость на кадре", fontsize=15)
    ax3.legend(fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(13, 8))
    ax4.plot(t_sum_reg, alpha_all_reg, '-', c='grey', label='True')
    ax4.plot(t_sum_reg, alpha_all_savg, '-', c='b', label='savgol_filter, window_size=2001')
    ax4.grid()
    ax4.set_title('Зависимость яркости разрядов от времени', fontsize=15)
    ax4.set_xlabel("Номер кадра", fontsize=15)
    ax4.set_ylabel("Суммарная яркость на кадре", fontsize=15)
    ax4.legend(fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    plt.show()

    fig5, ax5 = plt.subplots(figsize=(13, 8))
    I = integ(alpha_all_reg, t_sum_reg)
    ax5.plot(t_sum_reg, I, '-', c='b', label='#1')
    ax5.axvline(t_sum_reg[-1], c='r', ls='-.', lw=1, label=f'$W_1 ={int(round(I[-1]))}, N frame = {int(t_sum_reg[-1])}$')
    ax5.axhline(I[-1], c='r', ls='-.', lw=1)
    ax5.grid()
    ax5.set_title('Суммарная яркость', fontsize=15)
    ax5.set_xlabel("Номер кадра", fontsize=15)
    ax5.set_ylabel("Суммарная яркость на предыдущих кадрах", fontsize=15)
    ax5.legend(loc = 2,  fontsize=15)
    ax5.tick_params(axis='both', labelsize=15)
    plt.show()