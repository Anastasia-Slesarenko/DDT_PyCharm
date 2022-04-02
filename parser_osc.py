from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd

if __name__ == '__main__':

   path_file_coord = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_18.02.22_part/CD/Oscillogram/'
   os.chdir(path_file_coord)
   mfiles = glob('*.m')
   I = []
   for fname in mfiles:
      with open(fname, 'r') as f:
         m_code = f.read()

      RealChannelsQuantity = int(m_code.split('RealChannelsQuantity')[1].split('= ')[1].split(';')[0])
      RealKadrsQuantity = int(m_code.split('RealKadrsQuantity')[1].split('= ')[1].split(';')[0])
      TotalTime = float(m_code.split('TotalTime')[1].split('= ')[1].split(';')[0])

      DataCalibrZeroK = m_code.split('DataCalibrZeroK')[1].split('= ')[1].split(';')[0].replace("[", "").replace("]",
                                                                                                                 "")
      DataCalibrZeroK = [float(i) for i in DataCalibrZeroK.split(' ') if i != '']

      DataCalibrScale = m_code.split('DataCalibrScale')[1].split('= ')[1].split(';')[0].replace("[", "").replace("]",
                                                                                                                 "")
      DataCalibrScale = [float(i) for i in DataCalibrScale.split(' ') if i != '']

      DataCalibrOffset = m_code.split('DataCalibrOffset')[1].split('= ')[1].split(';')[0].replace("[", "").replace("]",
                                                                                                                   "")
      DataCalibrOffset = [float(i) for i in DataCalibrOffset.split(' ') if i != '']

      with open(fname.split('.')[0] + '.dat', "rb") as f:
         f.seek(0)  # seek
         y = np.fromfile(f, dtype='int16')
      y = y.reshape((RealKadrsQuantity, RealChannelsQuantity)).T

      fig, ax = plt.subplots(nrows=2, figsize=(8, 10))
      for i in range(RealChannelsQuantity):
         t = np.linspace(0, TotalTime, RealKadrsQuantity)
         ind = np.where(t <= 12)[0][-1]
         ch_signal = (y[i, :] + DataCalibrZeroK[i]) * DataCalibrScale[i] + DataCalibrOffset[i]
         ch_signal = -ch_signal * 10
         ax[i].plot(t[:ind], ch_signal[:ind])
         ax[i].axhline(np.mean(ch_signal[:ind]), c='k', label=f'{np.mean(ch_signal[:ind])}')
         ax[i].legend()
         ax[i].set_xlabel('t,sec')
         ax[i].set_ylabel('I, мкА')

      ch_signal = (y[0, :] + DataCalibrZeroK[0]) * DataCalibrScale[0] + DataCalibrOffset[0]
      ch_signal = -ch_signal * 10
      I += [np.mean(ch_signal[:ind])]
      plt.show()


   print(I)
   i = pd.DataFrame()
   i['I, мкА'] = I
   i.to_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_18.02.22_part/CD/I_1.csv', index=False)
