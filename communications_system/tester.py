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

import numpy as np
a = np.random.randint(0, 5)
print(a)
