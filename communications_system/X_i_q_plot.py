import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

three_db = np.load('X_3db_i_q_arr.npy')
seven_db = np.load('X_7db_i_q_arr.npy')
fifteen = np.load('X_15db_i_q_arr.npy')

plt.scatter(three_db[:,0], three_db[:,1], c='r')
plt.scatter(seven_db[:,0], seven_db[:,1], c = 'b')
plt.scatter(fifteen[:,0], fifteen[:,1],  c ='green')
plt.grid()
plt.show()