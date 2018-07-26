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
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure()
plt._imread('BPSK_AE_2_2.png')
#plt.plot(img)
plt.show()