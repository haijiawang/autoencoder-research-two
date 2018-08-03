import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


snr = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
#ray33_7 = np.load('rayleigh_auto_3_3_7dB')
ray_bpsk = np.load('rayleigh_bpsk.npy')
rayleigh = np.load('R_rayleigh_auto_3_3_7dB.npy')

plt.plot(snr, (3 * ray_bpsk), linestyle = '-', marker='s', color='red', label='bpsk' )
plt.plot(snr, rayleigh, linestyle = '-', marker='*', color='blue', label='AE ')
plt.yscale('log')
plt.ylabel('BER')
plt.xlabel('SNR')
plt.grid()
plt.legend(loc='upper right', ncol=1)
plt.savefig('R_bpsk_rayleigh_rician')
plt.show()