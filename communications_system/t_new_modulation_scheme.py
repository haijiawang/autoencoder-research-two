import numpy as np
i_q_array = np.load('t_i_q_arr.npy')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

M = 8
R = 1.0
EbNo = 7
c = 1
#q =
dict_i_j = {000: [c*1,c*0], 001: [(c * np.cos(2 * np.pi / 7)), (c * np.sin(2 * np.pi / 7))],
            010: [(c * np.cos(4 * np.pi / 7)), (c * np.sin(4 * np.pi / 7)) ],
            011: [(c * np.cos(6 * np.pi / 7)), (c * np.sin(6 * np.pi / 7))],
            100: [(c * np.cos(8 * np.pi / 7)), (c * np.sin(8 * np.pi / 7))],
            101: [(c * np.cos(10 * np.pi / 7)), (c * np.sin(10 * np.pi / 7))],
            110: [(c * np.cos(12 * np.pi / 7)), (c * np.sin(12 * np.pi / 7))],
            111: [c*0, c*0]}


k = np.log2(M)
k = int(k)
print ('M:', M, 'k:', k)

# generating data of size N
N = 1000
label = np.random.randint(M, size=N)
binarr = {}
binarr = {0: 000, 1: 001, 2: 010, 3: 100, 4: 110, 5:011, 6:101, 7:111}
t = np.arange(0.0, 10, 0.01)
iarr = []
qarr = []
for x in label:
    i = dict_i_j[binarr[x]][0]
    q = dict_i_j[binarr[x]][1]
    iarr.append(i)
    qarr.append(q)

fo = 0.5
i_carrier = np.cos(2 * np.pi * fo * t)
q_carrier = np.sin(2 * np.pi * fo * t)
transsignal = (iarr * i_carrier) + (qarr * q_carrier)
#transsignal = q_carrier
plt.figure()
plt.xlabel('time')
plt.ylabel('amplitude')
plt.plot(t, transsignal)
plt.show(block = False)

'''
FORMING THE RECIEVED SIGNAL
'''


from numpy.random import rand, randn
'''
EbNoN = 10.0 ** (EbNo/10.0)
x = 2 * (rand(N) >= 0.5) - 1 
noise_std = 1 / np.sqrt(2 * EbNo)
y = x + noise_std * randn(N)
'''
'''
EbNoR = 10.0 ** (EbNo/10.0)
noise_std = 1 / np.sqrt(2 * EbNoR)
noise = np.random.normal(0, noise_std, N)
noisesignal = noise + transsignal
'''

EbNodB = 7
EbNo = 10.0 ** (EbNodB / 10.0)
noise_std = 1 / np.sqrt(2 * R * EbNo)
noise = noise_std * randn(N)
noisesignal = transsignal + noise
plt.figure()
plt.plot(t, noisesignal)
plt.show()

'''
DEMODULATE THE SIGNAL
'''
