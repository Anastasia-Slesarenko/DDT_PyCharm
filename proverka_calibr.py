import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # #param = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/Postresult/data_with_criterion/dat_t_W.csv')
    # datfil = ['PKD_02.02.22_part', 'PKD_02.02.22_part', 'PKD_09.02.22_part', 'PKD_10.02.22_part', 'PKD_11.02.22_part', 'PKD_18.02.22_part']
    # fil = 'PKD_18.02.22_part'
    # n = [1, 2, 4, 5, 8, 10]
    # fig2, ax2 = plt.subplots()
    # alpha_15 = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/alpha_corona_15.csv')
    # alpha_10 = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/alpha_corona_10.csv')
    # I_dat = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/I_1.csv', index_col=False)
    #
    # y_15 = alpha_15['alpha'].tolist()
    # y_10 = alpha_10['alpha'].tolist()
    # I = (I_dat['I, мкА'].iloc[1:] - I_dat['I, мкА'].iloc[0]).tolist()
    # fx_15 = np.linspace(np.min(y_15), np.max(y_15), 100)
    # fx_10 = np.linspace(np.min(y_10), np.max(y_10), 100)
    # fp_15, residulas_15, rank_15, sv_15, rcond_15 = np.polyfit(y_15, I, 1, full=True)
    # fp_10, residulas_10, rank_10, sv_10, rcond_10 = np.polyfit(y_10, I, 1, full=True)
    # f_15 = np.poly1d(fp_15)
    # f_10 = np.poly1d(fp_10)
    #
    # ax2.scatter(y_15, I, color='r')
    # ax2.plot(fx_15, f_15(fx_15), label=f'hold=15')
    #
    # ax2.scatter(y_10, I, color='b')
    # ax2.plot(fx_10, f_10(fx_10), label=f'hold=10')
    #
    # ax2.grid()
    # ax2.set_title('Ток коронного разряда от яркости')
    # ax2.set_xlabel("Яркость, усл. ед.")
    # ax2.set_ylabel("Ток, мкА")
    # ax2.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # y_10 = np.array(y_10)
    # y_14 = np.array(y_15)
    # p = np.abs(y_15-y_10)/y_15*100
    # print(p)
    # ax.scatter(list(15*np.ones(len(y_15))), y_15, color='r')
    # ax.scatter(list(10*np.ones(len(y_10))), y_10, color='b')
    # ax.grid()
    # ax.set_title('Зависимость яркости короны от порога')
    # ax.set_xlabel("Порог по яркости, усл. ед.")
    # ax.set_ylabel("Средняя яркость короны")
    # ax.legend()
    # plt.show()

    # Получить пороги для всех дней
    # Приводим все ко дню с максимальным порогом 15 от 18.02

    #param = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/Postresult/data_with_criterion/dat_t_W.csv')
    datfil = ['PKD_02.02.22_part', 'PKD_02.02.22_part', 'PKD_09.02.22_part', 'PKD_10.02.22_part', 'PKD_11.02.22_part', 'PKD_18.02.22_part']
    fil_id = 'PKD_18.02.22_part'
    n = [1, 2, 4, 5, 8, 10]
    n_p = [92, 92, 92, 104, 101, 117]
    fig2, ax2 = plt.subplots()
    hold = 15
    int_h = hold*117
    I_hold = []
    alpha_hold = []
    alpha_id = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{datfil[-1]}/CD/alpha_corona_15.csv')
    I_dat_id = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{datfil[-1]}/CD/I_1.csv', index_col=False)
    y_id = alpha_id['alpha'].tolist()
    I_id = I_dat_id['I, мкА'].iloc[1:].tolist()
    fx_id = np.linspace(np.min(y_id), np.max(y_id), 100)
    fp_id = np.polyfit(y_id, I_id, 1)
    f_id = np.poly1d(fp_id)
    I_id = f_id(int_h)
    c = (I_id-fp_id[1])/fp_id[0]/n_p[-1]
    print(fp_id)

    alpha = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{datfil[3]}/CD/alpha_corona_15.csv')
    I_dat = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{datfil[3]}/CD/I_1.csv', index_col=False)
    y = alpha['alpha'].tolist()
    I = I_dat['I, мкА'].iloc[1:].tolist()
    fp = np.polyfit(y, I, 1)
    f = np.poly1d(fp)
    ah = (I_id - fp[1]) / fp[0] / n_p[3]
    print(fp, ah)
    # h = ['10', '10', '10', '14', '13', '15']
    fig2, ax2 = plt.subplots()
    for fil, h, i in zip(datfil, n_p, n):
        alpha = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/alpha_corona_15.csv')
        I_dat = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/I_1.csv', index_col=False)

        y = alpha['alpha'].tolist()
        # I = (I_dat['I, мкА'].iloc[1:] - I_dat['I, мкА'].iloc[0]).tolist()
        I = I_dat['I, мкА'].iloc[1:].tolist()
        fx = np.linspace(np.min(y), np.max(y), 100)
        fp = np.polyfit(y, I, 1)
        f = np.poly1d(fp)
        alpha_hold += [(I_id-fp[1])/fp[0]/h]
        print(fp)
        ax2.scatter(y, I)
        ax2.plot(fx, f(fx), label=f'{i}')


    ax2.grid()
    ax2.set_title('Ток коронного разряда от яркости')
    ax2.set_xlabel("Яркость, усл. ед.")
    ax2.set_ylabel("Ток, мкА")
    ax2.legend()
    plt.show()
    # v = alpha_hold/np.array(n_p)

    print(alpha_hold)

