import random
import numpy as np

for i in range(0,10):
    ch_matrix = np.sqrt((np.random.randn(3) ** 2) + (np.random.randn(3) ** 2)) / np.sqrt(2.0)
    print(ch_matrix)
