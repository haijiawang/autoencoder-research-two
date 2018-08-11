import random
import numpy as np
import sklearn.preprocessing as pp
import numpy.linalg as LA

test = np.load('test.npy')
std = np.std(test)
print('initial', std)

test = pp.normalize(test)
std = np.std(test)
print('new', std)

