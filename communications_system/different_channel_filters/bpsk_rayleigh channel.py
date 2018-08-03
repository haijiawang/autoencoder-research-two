from numpy import sqrt
import random
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

N = 100000
EbNodB_range = list(np.arange(-10, 5, 0.5))
itr = len(EbNodB_range)
ber = [None] * itr
tx_symbol = 0
noise = 0
ch_coeff = 0
rx_symbol = 0
det_symbol = 0

for n in range(0, itr):

    EbNodB = EbNodB_range[n]
    EbNo = 10.0 ** (EbNodB / 10.0)
    noise_std = 1 / sqrt(2 * EbNo)
    noise_mean = 0

    no_errors = 0
    for m in range(0, N):
        tx_symbol = 2 * random.randint(0, 1) - 1
        noise = random.gauss(noise_mean, noise_std)
        ch_coeff = sqrt(random.gauss(0, 1) ** 2 + random.gauss(0, 1) ** 2) / sqrt(2)
        rx_symbol = tx_symbol * ch_coeff + noise
        det_symbol = 2 * (rx_symbol >= 0) - 1
        no_errors += 1 * (tx_symbol != det_symbol)

    ber[n] = 1.0 * no_errors / N
    print('SNR: ', EbNodB_range[n], 'BER: ', ber[n])

np.save('rayleigh_bpsk', ber)
plt.plot(EbNodB_range, ber, 'bo-')
# plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.grid()
plt.title('BPSK Modulation')
plt.show()