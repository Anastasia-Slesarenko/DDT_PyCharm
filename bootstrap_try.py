import numpy as np
from scipy.stats import bootstrap
import pandas as pd
from tqdm import tqdm
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
import scipy as sp
from scipy.optimize import fsolve

if __name__ == '__main__':
    param = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/Postresult/data_with_criterion/dat_t_W.csv')
    datfil = ['PKD_02.02.22_part', 'PKD_02.02.22_part', 'PKD_09.02.22_part', 'PKD_10.02.22_part', 'PKD_11.02.22_part', 'PKD_18.02.22_part']
    W_all = param['W']
    I_w = []
    n = [1, 2, 4, 5, 8, 10]
    fig2, ax2 = plt.subplots()
    for fil, W, i in zip(datfil, W_all, n):
        alpha = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/alpha_corona.csv')
        I_dat = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/I_1.csv', index_col=False)

        y = alpha['alpha'].tolist()
        I = (I_dat['I, мкА'].iloc[1:] - I_dat['I, мкА'].iloc[0]).tolist()
        # I = I_dat['I, мкА'].iloc[1:].tolist()
        fx = np.linspace(np.min(y), np.max(y), 100)
        fp, residulas, rank, sv, rcond = np.polyfit(y, I, 1, full=True)
        f = np.poly1d(fp)
        print('Параметры модели: %s' % fp)
        print(f(W))
        I_w += [f(W)]
        ax2.scatter(y, I)
        ax2.plot(fx, f(fx), label=f'{i}')

    ax2.grid()
    ax2.set_title('Ток коронного разряда от яркости')
    ax2.set_xlabel("Яркость, усл. ед.")
    ax2.set_ylabel("Ток, мкА")
    ax2.legend()
    plt.show()

    print(I_w)
    dat_I_t = pd.DataFrame()
    dat_I_t['I'] = I_w
    dat_I_t['t'] = param['t']
    dat_I_t.to_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/Postresult/data_with_criterion/dat_I_t.csv', index=False)

    fig1, ax1 = plt.subplots()
    plot_bb = pd.DataFrame()
    plot_bb['data'] = (dat_I_t['t'] / dat_I_t['t'].median()).tolist() + (dat_I_t['I'] / dat_I_t['I'].median()).tolist()\
                      + (param['W'] / param['W'].median()).tolist()

    plot_bb['variable'] = ['Time']*dat_I_t['t'].shape[0] + ['Current']*dat_I_t['I'].shape[0] + ['Brightness']*param['W'].shape[0]
    sns.boxplot(data=plot_bb, x='variable', y='data', ax=ax1)
    sns.swarmplot(data=plot_bb, x='variable', y='data', ax=ax1, color='r', alpha=0.7)
    ax1.grid()
    #plt.show()

    fig, ax = plt.subplots()
    ax.scatter(param['W'], dat_I_t['I'])
    ax.grid()
    plt.show()