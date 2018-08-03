import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

custom_auto = np.load('R_custom_autoencoder33.npy')
orig_auto = np.load('R_orignal_autoencoder33.npy')
rician = np.load('R_rician_autoencoder33.npy')
snr = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] #5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
#plt.plot(snr, custom_auto, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'Original AE (3,3)')
#plt.plot(snr, orig_auto, linestyle = '-', marker = 's', color = 'blue', label = 'Custom AE (3,3)')
plt.plot(snr, rician, linestyle='-', marker='s', color='blue', label='Rician')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('R_original_autoencoder')
#plt.show()

ch_matrix = np.sqrt((np.random.randn(3) ** 2) + (np.random.randn(3) ** 2)) / np.sqrt(2.0)

print(ch_matrix)