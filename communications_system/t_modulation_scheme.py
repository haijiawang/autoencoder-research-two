import commpy
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

'Mapping Binary Digits to Constellation Points'
c = 1
dict_i_j = {000: [c*1,c*0], 001: [(c * np.cos(2 * np.pi / 7)), (c * np.sin(2 * np.pi / 7))],
            010: [(c * np.cos(4 * np.pi / 7)), (c * np.sin(4 * np.pi / 7)) ],
            011: [(c * np.cos(6 * np.pi / 7)), (c * np.sin(6 * np.pi / 7))],
            100: [(c * np.cos(8 * np.pi / 7)), (c * np.sin(8 * np.pi / 7))],
            101: [(c * np.cos(10 * np.pi / 7)), (c * np.sin(10 * np.pi / 7))],
            110: [(c * np.cos(12 * np.pi / 7)), (c * np.sin(12 * np.pi / 7))],
            111: [c*0, c*0]}


temp = dict_i_j.values()
i_arr = []
j_arr = []
for i in temp:
    i_arr.append(i[0])
print(i_arr)

for j in temp:
    j_arr.append(j[1])
print(j_arr)

plt.scatter(i_arr, j_arr)
plt.grid()
#plt.axis((-1.5,1.5,-1.5,1.5))
plt.show()
