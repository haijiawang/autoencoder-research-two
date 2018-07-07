import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from numpy.lib.stride_tricks import as_strided

from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train_left, y_train_right), (X_test_left, y_test_right) = mnist.load_data()
X_train_left = X_train_left[:, :, :14]

plt.subplot(221)
plt.imshow(X_train_left[0], cmap=plt.get_cmap('gray'))
plt.show()
