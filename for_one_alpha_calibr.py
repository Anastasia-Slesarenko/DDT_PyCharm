import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':
    datfil = ['PKD_02.02.22_part', 'PKD_02.02.22_part', 'PKD_09.02.22_part', 'PKD_10.02.22_part', 'PKD_11.02.22_part', 'PKD_18.02.22_part']
    fil_id = 'PKD_18.02.22_part'
    n = [1, 2, 4, 5, 8, 10]
    n_p = [9, 9, 14, 12, 12, 15]
    a = []
    b = []

    fig2, ax2 = plt.subplots()
    for fil, i in zip(datfil, n):
        alpha = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/alpha_corona_15.csv')
        I_dat = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/CD/I_1.csv', index_col=False)
        y = alpha['alpha'].tolist()
        I = I_dat['I, мкА'].iloc[1:].tolist()
        fx = np.linspace(0, np.max(y), 100)
        fp = np.polyfit(y, I, 1)
        f = np.poly1d(fp)
        a += [fp[0]]
        b += [fp[1]]
        if fil == 'PKD_02.02.22_part':
            a_m = np.max(y)
            I1 = f(a_m)
        if fil == 'PKD_18.02.22_part':
            a10 = (I1 - fp[1]) / fp[0]
        ax2.scatter(y, I)
        ax2.plot(fx, f(fx), label=f'{i}')
    c = (a10 - a_m) / a_m * 100
    print(a10, a_m, c)
    ax2.grid()
    ax2.set_title('Ток коронного разряда от яркости')
    ax2.set_xlabel("Яркость, усл. ед.")
    ax2.set_ylabel("Ток, мкА")
    ax2.legend()
    plt.show()
    a_10 = a[-1]
    b_10 = b[-1]
    a = np.array(a)
    b = np.array(b)
    #
    # for a, b, fil, n, h in tqdm(zip(a, b, datfil, n, n_p)):
    #     data = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/sum_df_new.cvs')
    #     data_reg = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/reg_data_new.cvs')
    #     data_not_reg = pd.read_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/not_reg_data_new.cvs')
    #
    #
    #     alpha_data = data['alpha_sum'].values
    #     alpha_new = (alpha_data*a + b - b_10)/a_10
    #     data['alpha_new'] = alpha_new
    #     data.to_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/sum_df_new.cvs', index=False)
    #
    #     alpha_data_not_reg = data_not_reg['alpha_all'].values
    #     alpha_new_not_reg = (alpha_data_not_reg * a + b - b_10) / a_10
    #     data_not_reg['alpha_new'] = alpha_new_not_reg
    #     data_not_reg.to_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/not_reg_data_new.cvs', index=False)
    #
    #     alpha_data_reg = data_reg['alpha_all'].values
    #     ind = data_reg[data_reg['alpha_all'] == h].index.tolist()
    #     alpha_new_reg = (alpha_data_reg * a + b - b_10) / a_10
    #     data_reg['alpha_new'] = alpha_new_reg
    #     for i in ind:
    #         data_reg.loc[data_reg['alpha_new'] == data_reg['alpha_new'][i], 'alpha_new'] =0
    #     data_reg.to_csv(f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/{fil}/{n}/test_frames/cap_cut_crop.mp4/reg_data_new.cvs', index=False)




