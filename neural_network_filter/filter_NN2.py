from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import pylab
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

#DIVIDE INTO BLURRY/SHARP CATEGORIES
blurry_train = np.copy(x_train)
sharp_train = np.copy(x_train)
blurry_test = np.copy(x_test)
sharp_test = np.copy(x_test)

#BLURRY IMAGES
for i in range(60000):
    blurry_train[i] = gaussian_filter(blurry_train[i], sigma=1)

for i in range(10000):
    blurry_test[i] = gaussian_filter(blurry_test[i], sigma=1)


#SHARP IMAGES
alpha = 100
for i in range(10):
    blurred = gaussian_filter(sharp_train[i], sigma=1)
    sharp_train[i] = sharp_train[i] + alpha * (sharp_train[i] - blurred)

n = 10
plt.figure(figsize=(20,6))
for i in range(n):
    #original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #blurred
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(blurry_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #sharp
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(sharp_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()