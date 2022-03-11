from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from predictr import Analysis, PlotAll

def integ(I, t):
    I_s = []
    for t_i in t:
        ind = int(np.where(t == t_i)[0])
        I_s += [np.trapz(I[:ind], dx=1)]
    return I_s

if __name__ == '__main__':

    path_file_coord = 'C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/Postresult/'
    os.chdir(path_file_coord)
    result = glob.glob('*.cvs')
    n_f = []
    W_1 = []

    fig, ax = plt.subplots(figsize=(13, 8))
    for n in tqdm(sorted(result)):
        number = int(re.findall(r'\d+', n)[0])
        d = pd.read_csv(n, index_col=0)
        ax.semilogy(d['t']/60, d['I'], '-', label=f'{number}')
        n_f += [d['t'].iloc[-1]/60]
        W_1 += [d['I'].iloc[-1]]
        # ax.axvline(d['t'].iloc[-1], ls='-.', lw=1,
        #            label=f'$W_{number} ={int(round(W_1))}, N frame = {int(n_f)}$')
        # ax.axhline(d['I'].iloc[-1], ls='-.', lw=1)
        ax.legend(loc=2, fontsize=15)
    ax.grid()
    ax.set_title('Суммарная яркость', fontsize=15)
    ax.set_xlabel("Время, мин", fontsize=15)
    ax.set_ylabel("Суммарная яркость на предыдущих кадрах, усл. ед.", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    # plt.show()

    data = pd.DataFrame()
    data['W'] = W_1
    data['t'] = n_f
    # sns_plot = sns.histplot(data['t'])
    # fig2 = sns_plot.get_figure()
    # plt.show()

    # x_W = Analysis(df=data['W'], show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Brightness', unit='a.u.')
    # x_t = Analysis(df=data['t'], show=True, bounds='lrb', bounds_type='2s', cl=0.95, unit='min')

    # Brightness = Analysis(df=data['W']/data['W'].mean(), show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Failure', unit='a.u.')
    # Brightness.mle()
    # Time = Analysis(df=data['t']/data['t'].mean(), show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Failure',  unit='a.u.')
    # Time.mle()
    # objects = {fr'Brightness: $\widehat\beta$={Brightness.beta:4f} | $\widehat\eta$={Brightness.eta:4f}': Brightness,
    #            fr'Time: $\widehat\beta$={Time.beta:4f} | $\widehat\eta$={Time.eta:4f}': Time}
    # PlotAll(objects).mult_weibull()

    fig2, ax2 = plt.subplots()

    # df_std.melt(var_name='Column', value_name='Normalized')
    plot_bb = pd.DataFrame()
    plot_bb['data'] = (data['t'] / data['t'].mean()).tolist() + (data['W'] / data['W'].mean()).tolist()
    plot_bb['variable'] = ['time']*data['t'].shape[0] + ['brightness']*data['W'].shape[0]

    # plot_bb['data'] = ((data['t']-data['t'].mean())/data['t'].std()).tolist() + ((data['W']-data['W'].mean())/data['W'].std()).tolist()
    # plot_bb['variable'] = ['time']*data['t'].shape[0] + ['brightness']*data['W'].shape[0]

    # plot_bb['data'] = ((data['t']-data['t'].min())/(data['t'].max()-data['t'].min())).tolist() + ((data['W']-data['W'].min())/(data['W'].max()-data['W'].min())).tolist()
    # plot_bb['variable'] = ['time']*data['t'].shape[0] + ['brightness']*data['W'].shape[0]

    sns.boxplot(data=plot_bb, x='variable', y='data', ax=ax2)
    ax2.grid()
    plt.show()




    # for n in tqdm(sorted(result)):
    #     number = int(re.findall(r'\d+', n)[0])
    #     d = pd.read_csv(n, index_col=0)
    #     alpha_all_reg = np.array(d['alpha_all'])
    #     if number == 9:
    #         t_sum_reg = np.array(d['t_sum'] / 30)
    #     else:
    #         t_sum_reg = np.array(d['t_sum']/25)
    #     Int = integ(alpha_all_reg, t_sum_reg)
    #     reg_data = pd.DataFrame()
    #     reg_data['I'] = Int
    #     reg_data['t'] = t_sum_reg
    #     reg_data.to_csv(f'data_{number}.cvs')