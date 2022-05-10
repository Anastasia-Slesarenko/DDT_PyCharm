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

    path_file_coord = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/Postresult/data_with_criterion/'
    os.chdir(path_file_coord)
    result = glob.glob('*.cvs')
    n_f = []
    W_1 = []
    print(result[0])
    i = [1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(10, 7))
    for n, k in zip(sorted(result), i):
        number = int(re.findall(r'\d+', n)[0])
        d = pd.read_csv(n, index_col=0)
        ax.plot(d['t_sum']/60, d['I'], '-', linewidth=2, label=f'{k}')
        #ax.plot(d['t_sum'] / 60, d['I'], '-', label=f'{number}')
        n_f += [d['t_sum'].iloc[-1]/60]
        W_1 += [d['I'].iloc[-1]]
        # ax.axvline(d['t'].iloc[-1], ls='-.', lw=1,
        #            label=f'$W_{number} ={int(round(W_1))}, N frame = {int(n_f)}$')
        # ax.axhline(d['I'].iloc[-1], ls='-.', lw=1)
        ax.legend(loc=2, fontsize=15)
    ax.grid()
    #ax.set_title('Суммарная яркость', fontsize=15)
    ax.set_xlabel("Time, (min)", fontsize=15)
    ax.set_ylabel("Accumulated intensity of the PD light, (a.u.)", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.yaxis.get_offset_text().set_size(15)
    plt.show()

    n_f = np.array(n_f)
    W_1 = np.array(W_1)
    d_t = (np.max(n_f)-np.min(n_f))/np.mean(n_f)*100
    d_W = (np.max(W_1) - np.min(W_1)) / np.mean(W_1)*100
    print(d_t, d_W)

    # data = pd.DataFrame()
    # data['W'] = W_1
    # data['t'] = n_f
    # fig3, ax3 = plt.subplots()
    # ax3.scatter(data['t'] / data['t'].median(), data['W'] / data['W'].median())
    #
    # #data.to_csv(r'C:\Users\1\Desktop\VKR\Master\data\silicone SPBU\Postresult\data_with_criterion\dat_t_W.csv', index=False)
    #
    # ax3.grid()
    # ax3.set_xlabel("Время")
    # ax3.set_ylabel("Суммарная яркость к концу теста")
    # ax3.tick_params(axis='both')
    # plt.show()
    #
    # fig5, ax5 = plt.subplots()
    # ax5.scatter([20.3, 21.2, 21.7, 22.5, 22.2], data['W'][1:])
    # ax5.grid()
    # ax5.set_xlabel("Температура, С", fontsize=15)
    # ax5.set_ylabel("Суммарная яркость к концу теста, усл. ед.", fontsize=15)
    # ax5.tick_params(axis='both', labelsize=15)
    # plt.show()
    #
    # fig2, ax2 = plt.subplots()
    # plot_bb = pd.DataFrame()
    # plot_bb['data'] = (data['t'] / data['t'].mean()).tolist() + (data['W'] / data['W'].mean()).tolist()
    # plot_bb['variable'] = ['time']*data['t'].shape[0] + ['brightness']*data['W'].shape[0]
    #
    # # plot_bb['data'] = ((data['t']-data['t'].mean())/data['t'].std()).tolist() + ((data['W']-data['W'].mean())/data['W'].std()).tolist()
    # # plot_bb['variable'] = ['time']*data['t'].shape[0] + ['brightness']*data['W'].shape[0]
    #
    # sns.boxplot(data=plot_bb, x='variable', y='data', ax=ax2)
    # #sns.stripplot(data=plot_bb, x='variable', y='data', ax=ax2, color='r', alpha=0.7)
    # #sns.violinplot(data=plot_bb, x='variable', y='data', ax=ax2)
    # sns.swarmplot(data=plot_bb, x='variable', y='data', ax=ax2, color='r', alpha=0.7)
    # ax2.grid()
    # plt.show()
    #
    # std_t = (data['t'] / data['t'].median()).std()
    # std_W = (data['W'] / data['W'].median()).std()
    #
    # x_W = Analysis(df=data['W'], show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Brightness', unit='a.u.')
    # x_t = Analysis(df=data['t'], show=True, bounds='lrb', bounds_type='2s', cl=0.95, unit='min')
    #
    # Brightness = Analysis(df=data['W']/data['W'].median(), show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Brightness of Failure', unit='a.u.')
    # Brightness.mle()
    # Time = Analysis(df=data['t']/data['t'].median(), show=True, bounds='lrb', bounds_type='2s', cl=0.95, x_label='Time of Failure',  unit='a.u.')
    # Time.mle()
    # objects = {fr'Brightness: $\widehat\beta$={Brightness.beta:4f} | $\widehat\eta$={Brightness.eta:4f}': Brightness,
    #            fr'Time: $\widehat\beta$={Time.beta:4f} | $\widehat\eta$={Time.eta:4f}': Time}
    # PlotAll(objects).mult_weibull(x_label='Moment of Failure')

    #bounds ='mcpb'





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