import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

snr2_array = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
snr_array = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
auto_3 = np.load('auto_range10_10_20.npy')
test = np.load('auto_3_3.npy')
print(auto_3.shape)
#auto_db4 = np.load('auto_range_-4.npy')
#auto_db6 = np.load('auto_range_-6.npy')
#auto_db8 = np.load('auto_range_-8.npy')
#auto_db10 = np.load('auto_range_10.npy')
#auto_db12 = np.load('auto_range_12.npy')
#auto_db14 = np.load('auto_range_14.npy')

plt.plot(snr2_array, auto_3, linestyle = ':', marker = 's', color = 'darkseagreen', label = 'Train over Range')
plt.plot(snr2_array, test, linestyle = '-', marker = '*', color = 'darkgrey', label = 'Original (3,3)')
#plt.plot(snr_array, auto_db6, linestyle = ':', marker = 's', color = 'cyan', label = '6')
#plt.plot(snr_array, auto_db8, linestyle = '-', marker = '*', color = 'greenyellow', label = '8')

#plt.plot(snr_array, auto_db10, linestyle = ':', marker = 's', color = 'teal', label = '10')
#plt.plot(snr_array, auto_db12, linestyle = '-', marker = '*', color = 'darkseagreen', label = '12')

#plt.plot(snr_array, auto_db14, linestyle = ':', marker = 's', color = 'red', label = '14')

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('AE_RANGE_TEST')
plt.show()