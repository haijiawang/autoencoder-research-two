'''
# generating data of size N
import numpy as np
M = 100
N = 10000
label = np.random.randint(100, size=N)

vec_dim = int(np.log2(M))
print(vec_dim)

# creating binary vectors
data = []
for i in label:
    temp = np.zeros(vec_dim)
    num = i
    num = np.binary_repr(num)
    #num = int (num)
    temp_list = map(int, str(num))
    temp_list = np.asarray(temp_list)
    #print(temp_list)
    #temp = np.append(temp_list, temp)
    for x in range(0, len(temp_list)):
       temp[x] = temp_list[x]
    data.append(temp_list)

data = np.array(data)
for i in range(0, 10):
    print(data[i])
'''
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = [1, 2, 0, 3]
y = [3,1,2,3]
z = [1, 1,1, 1]
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(x, y, z)
plt.show()
'''
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#define time
t = np.arange(0.0, 2, 0.01)
freq = 10 #carrier frequency
freqs = 1 #signal frequency
m = 0.3
carrier = (1 + (m * np.sin(2 * np.pi * freqs *t))) * (np.sin(2 * np.pi * freq * t))
#plt.xlabel('time')
#plt.ylabel('amplitude')
#plt.plot(t, carrier)
#plt.show()
print(carrier.shape)
#plt.scatter(carrier)
#plt.show()
'''
'''
import sys
from qam import Qam

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
	print "Usage: %s <data-bits> <data-bits>" % sys.argv[0]
	exit(1)

modulation = {
    '0000' : (1.4142, 135.0000),
    '0001' : (1.1180, 116.5650),
    '0010' : (1.4142,  45.0000),
    '0011' : (1.1180,  63.4350),
    '0100' : (1.4142, 225.0000),
    '0101' : (1.1180, 243.4350),
    '0110' : (1.4142, 315.0000),
    '0111' : (1.1180, 296.5650),
    '1000' : (1.1180, 153.4350),
    '1001' : (0.7071, 135.0000),
    '1010' : (1.1180,  26.5650),
    '1011' : (0.7071,  45.0000),
    '1100' : (1.1180, 206.5650),
    '1101' : (0.7071, 225.0000),
    '1110' : (1.1180, 333.4350),
    '1111' : (0.7071, 315.0000),
}

q1 = Qam(baud_rate = 10,
         bits_per_baud = 4,
         carrier_freq = 10e3,
         modulation = modulation)

q2 = Qam(baud_rate = 10,
         bits_per_baud = 4,
         carrier_freq = 9.9e3,
         modulation = modulation)

s = q1.generate_signal(sys.argv[1]) + q2.generate_signal(sys.argv[2])

plt.figure(1)
#q.plot_constellation()
plt.figure(2)
s.plot(dB=False, phase=False, stem=False, frange=(0,12e3))
s.show()
'''
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

orig = np.load('R_orignal_autoencoder.npy')
red = np.load('R_reduced_autoencoder.npy')
SNR = snr_array = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
orig4 = np.load('R_orignal_autoencoder4.npy')
red4 = np.load('R_reduced_autoencoder4.npy')
orig5 = np.load('R_orignal_autoencoder5.npy')
red5 = np.load('R_reduced_autoencoder5.npy')
orig6 = np.load('R_custom_autoencoder6.npy')
red6 = np.load('R_reduced_autoencoder6.npy')

plt.figure()
plt.plot(SNR, orig, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'Original AE (3,3)')
plt.plot(SNR, red, linestyle='-', marker = 's', color = 'blue', label='Reduced AE (3,3)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)
plt.savefig('R_comparison33')
plt.show(block=False)

plt.figure()
plt.plot(SNR, orig4, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'Original AE (4,4)')
plt.plot(SNR, red4, linestyle='-', marker = 's', color = 'blue', label='Reduced AE (4,4)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)
plt.savefig('R_comparison44')
plt.show(block=False)

plt.figure()
plt.plot(SNR, orig5, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'Original AE (5,5)')
plt.plot(SNR, red5, linestyle='-', marker = 's', color = 'blue', label='Reduced AE (5,5)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)
plt.savefig('R_comparison55')
plt.show(block=False)

plt.figure()
plt.plot(SNR, orig6, linestyle = '-', marker = '*', color = 'darkseagreen', label = 'Original AE (6,6)')
plt.plot(SNR, red6, linestyle='-', marker = 's', color = 'blue', label='Reduced AE (6,6)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)
plt.savefig('R_comparison66')
plt.show(block=False)
'''
'''
import numpy as np
N = 10000
k = 3
M = 8
data = []
for i in range(10000):
    arr = np.zeros(k + 1)
    num = np.random.randint(0, M)
    num = (np.binary_repr(num))
    for j in range(0, len(num)):
        arr[j] = num[j]

    data.append(arr)


for i in range(0, 10):
    print(data[i])
'''
import random
z = random.normalvariate(0,1)
print(z)
