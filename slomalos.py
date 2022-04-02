import matplotlib.pyplot as plt
import matplotlib as mlt
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

    n_sample = 2
    d_f = 'PKD_02.02.22_part'

    reg_data = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{d_f}/{n_sample}/test_frames'
                           '/cap_cut_crop.mp4/reg_data.cvs', index_col=0)

    not_reg_data = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{d_f}/{n_sample}/test_frames'
                           '/cap_cut_crop.mp4/not_reg_data.cvs', index_col=0)

    sum_df = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{d_f}/{n_sample}/test_frames'
                         '/cap_cut_crop.mp4/sum_df.cvs', index_col=0)

    step_t_sum = np.diff(not_reg_data['t_sum'])

    ffp = 30 #30 #25
    t = 10
    num_el = ffp*t
    i = 0

    while i <= len(step_t_sum):
        if (step_t_sum[i:i + num_el] == 1).all():
            break
        else:
            i += 1

    # print(i+num_el, len(step_t_sum))
    # print(not_reg_data['t_sum'].shape)

    t_end = not_reg_data['t_sum'].iloc[i+num_el]
    t_start = not_reg_data['t_sum'].iloc[i]


    # sum_df['size'] = sum_df['alpha_sum'] s=sum_df['size'] cmap='hot'
    # sum_df.loc[sum_df['size'] <= 100, 'size'] = 0.5
    # sum_df.loc[sum_df['size'] > 100, 'size'] = 2.5
    sum_df_n = sum_df.groupby(['t_sum', 'y_sum'], as_index=False)['alpha_sum'].sum()
    sum_df_n['size'] = sum_df_n['alpha_sum']
    sum_df_n.loc[sum_df_n['size'] <= 100, 'size'] = 0.5
    sum_df_n.loc[sum_df_n['size'] > 100, 'size'] = 2.5

    sum_df_n['color'] = sum_df_n['alpha_sum']
    sum_df_n.loc[sum_df_n['color'] >= 255, 'color'] = 255
    # fig0, ax0 = plt.subplots()
    # sum_df_n[sum_df_n['alpha_sum'] <= 1000]['alpha_sum'].hist(ax=ax0)
    # plt.show()

    fig1, ax1 = plt.subplots(figsize=(15, 8))
    lc = ax1.scatter(sum_df_n['t_sum'], sum_df_n['y_sum'], s=sum_df_n['size'], c=sum_df_n['color'] , marker=',', cmap='hot')
    ax1.invert_yaxis()
    ax1.set_facecolor('silver')
    ax1.grid()
    ax1.axvline(t_end, c='grey', ls='-.', lw=1.5, label=f'в течение 10 секунды горит разряд')
    t = np.round(ax1.get_xticks() / 25 / 60)
    #t = [t.min(), t.min()*1.5, t.min()*2, t.mean(), t.mean()*2, t.max()]
    ax2 = ax1.twiny()

    ax2.set_xticks(t)
    ax2.set_xticklabels(t)
    ax2.set_xlabel('Time, min', fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)

    #ax1.set_title('Зависимость разрядов вдоль y-координаты от времени', fontsize=15)
    ax1.set_xlabel("Number of frame", fontsize=15)
    ax1.set_ylabel("y-coordinate", fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)
    lc.set_array(sum_df['alpha_sum'])
    lin = ax1.add_collection(lc)
    fig1.colorbar(lin, ax=ax1).ax.tick_params(labelsize=15)
    plt.rcParams.update({'font.size': 15})
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
    ax2.xaxis.get_offset_text().set_size(15)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(13, 8))
    alpha_all_reg = np.array(reg_data['alpha_all'])
    t_sum_reg = np.array(reg_data['t_sum'])
    alpha_all_savg = savgol_filter(alpha_all_reg, 2001, 1)
    ax3.semilogy(t_sum_reg, alpha_all_reg, '-', c='grey', label='True')
    ax3.semilogy(t_sum_reg, alpha_all_savg, '-', c='b', label='savgol_filter, window_size=2001')
    ax3.axvline(t_end, c='r', ls='-.', lw=1.5, label=f'в течение 10 секунды горит разряд')
    ax3.axvline(t_start, c='k', ls='-.', lw=1.5)
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
    ax4.axvline(t_end, c='r', ls='-.', lw=1.5, label=f'в течение 10 секунды горит разряд')
    ax4.axvline(t_start, c='k', ls='-.', lw=1.5)
    ax4.grid()
    ax4.set_title('Зависимость яркости разрядов от времени', fontsize=15)
    ax4.set_xlabel("Номер кадра", fontsize=15)
    ax4.set_ylabel("Суммарная яркость на кадре", fontsize=15)
    ax4.legend(fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    plt.show()
    #
    ind = int(np.where(t_sum_reg == not_reg_data['t_sum'].iloc[i+num_el])[0][0])
    # print(not_reg_data['t_sum'].iloc[i+num_el], t_sum_reg[ind])

    fig5, ax5 = plt.subplots(figsize=(13, 8))
    I1 = integ(alpha_all_reg, t_sum_reg)
    ax5.plot(t_sum_reg, I1, '-', c='b', label=f'{n_sample}')
    ax5.axvline(t_end, c='r', ls='-.', lw=1.5, label=f'в течение 10 секунды горит разряд')
    ax5.axvline(t_start, c='k', ls='-.', lw=1.5)
    ax5.axhline(I1[ind], c='g', ls='-.', lw=1, label=f'$W_1 ={int(round(I1[ind]))}, N frame = {int(t_end)}$')
    ax5.grid()
    ax5.set_title('Суммарная яркость', fontsize=15)
    ax5.set_xlabel("Номер кадра", fontsize=15)
    ax5.set_ylabel("Суммарная яркость на предыдущих кадрах", fontsize=15)
    ax5.legend(loc=2,  fontsize=15)
    ax5.tick_params(axis='both', labelsize=15)
    v = ax5.yaxis.get_offset_text()
    v.set_size(15)
    plt.show()

    # data_with_criterion = pd.DataFrame()
    # data_with_criterion['I'] = I1[:ind+1]
    # data_with_criterion['t_sum'] = t_sum_reg[:ind+1]/ffp
    # data_with_criterion.to_csv(f'C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/Postresult/data_with_criterion/'
    #                            f'data_with_criterion{n_sample}.cvs')
    print(alpha_all_savg[ind])


