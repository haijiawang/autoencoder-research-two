import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

snr2_array = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
snr_array = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
auto_3 = np.load('R_orignal_autoencoder3.npy')

bpsk33 = [1.964162538, 1.907114458, 1.847462824, 1.785187191, 1.720287165, 1.652786407, 1.582737032, 1.510224407, 1.435372265, 1.358348014, 1.27936811, 1.198703221, 1.116682905, 1.033699384, 0.9502099129, 0.8667371636, 0.7838669311, 0.7022424484, 0.6225545757, 0.545527231, 0.4718976212, 0.40239119, 0.3376917119, 0.2784076557, 0.2250367702, 0.1779317258, 0.1372704454, 0.1030352501, 0.07500490824, 0.05276286318, 0.03572320289, 0.02317338986, 0.01432974469, 0.008398829037, 0.004636048892, 0.002392778011, 0.001145446644, 0.0005039972351, 0.0002017633705, 0.00007265335966, 0.00002323264929, 0.00000650309053, 0.000001567840772, 0.0000003197198422, 0.0000000540360621, 0.00000000739817068, 0.0000000007997586105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
bpsk33 = np.asarray(bpsk33)
#bpsk_orig = [0.654720846, 0.6357048192, 0.6158209414, 0.5950623969, 0.5734290552, 0.5509288023, 0.5275790105, 0.5034081358	0.4784574215	0.4527826715	0.4264560367	0.3995677402	0.372227635	0.3445664612	0.3167366376	0.2889123879	0.261288977	0.2340808161	0.2075181919	0.1818424103	0.1572992071	0.1341303967	0.112563904	0.09280255191	0.07501225672	0.05931057525	0.04575681512	0.03434508336	0.02500163608	0.01758762106	0.0119077343	0.007724463286	0.004776581562	0.002799609679	0.001545349631	0.0007975926703	0.0003818155482	0.0001679990784	6.73E-05	2.42E-05	7.74E-06	2.17E-06	5.23E-07	1.07E-07	1.80E-08	2.47E-09	2.67E-10	2.21E-11	1.36E-12	6.01E-14	1.82E-15	3.64E-17	4.53E-19	3.33E-21	1.35E-23	2.82E-26	2.79E-29	1.20E-32	2.00E-36	1.17E-40	2.09E-45]
print(auto_3.shape)
#auto_db4 = np.load('auto_range_-4.npy')
#auto_db6 = np.load('auto_range_-6.npy')
#auto_db8 = np.load('auto_range_-8.npy')
#auto_db10 = np.load('auto_range_10.npy')
#auto_db12 = np.load('auto_range_12.npy')
#auto_db14 = np.load('auto_range_14.npy')

plt.plot(snr2_array, auto_3, linestyle = ':', marker = 's', color = 'darkseagreen', label = 'AE (3,3)')
plt.plot(snr2_array, bpsk33 / 3 , linestyle = ':', marker = 's', color = 'cyan', label = 'Original BPSK')
plt.plot(snr2_array, bpsk33, linestyle = '-', marker = '*', color = 'darkgrey', label = 'BPSK (3,3)')
#plt.plot(snr_array, auto_db6, linestyle = ':', marker = 's', color = 'cyan', label = '6')
#plt.plot(snr_array, auto_db8, linestyle = '-', marker = '*', color = 'greenyellow', label = '8')

#plt.plot(snr_array, auto_db10, linestyle = ':', marker = 's', color = 'teal', label = '10')
#plt.plot(snr_array, auto_db12, linestyle = '-', marker = '*', color = 'darkseagreen', label = '12')

#plt.plot(snr_array, auto_db14, linestyle = ':', marker = 's', color = 'red', label = '14')

plt.yscale('log')
plt.ylim(ymin=(4.32 * (10 ** -6)))
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('AE_RANGE_TEST')
plt.show()