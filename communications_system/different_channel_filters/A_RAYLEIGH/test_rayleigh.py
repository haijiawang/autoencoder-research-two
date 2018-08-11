
# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import random as rn
from rayleigh_custom_layers import Rayleigh, ch_matrix, BatchNormalization1, CustomNormalization
import random
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io



# defining parameters
M = 8
k = np.log2(M)
k = int(k)
print ('M:', M, 'k:', k)

# generating data of size N
N = 10000
label = np.random.randint(M, size=N)

# creating one hot encoded vectors# creati
data = []
for i in label:
    temp = np.zeros(M)
    temp[i] = 1
    data.append(temp)

data = np.array(data)

R = 3.0 / 3.0
n_channel = 3
input_signal = Input(shape=(M,))
encoded = Dense(M, activation='relu')(input_signal)
encoded1 = Dense(n_channel, activation='linear')(encoded)
stddev= 1.0
encoded2 = CustomNormalization(stddev)(encoded1)

n = 7
EbNo_train = np.power(10, n/10.0)  # coverted 7 db of EbNo
alpha1 = pow((2 * R * EbNo_train), -0.5)
encoded3 = Rayleigh(alpha1)(encoded2)

decoded = Dense(M, activation='relu')(encoded3)
decoded1 = Dense(M, activation='softmax')(decoded)

autoencoder = Model(input_signal, decoded1)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

autoencoder.summary()

N_val = 1500
val_label = np.random.randint(M, size=N_val)
val_data = []
for i in val_label:
    temp = np.zeros(M)
    temp[i] = 1
    val_data.append(temp)
val_data = np.array(val_data)

autoencoder.fit(data, data,
                epochs=100,
                batch_size=300,
                verbose=2,
                validation_data=(val_data, val_data))

from keras.models import load_model

encoder = Model(input_signal, encoded2)

encoded_input = Input(shape=(n_channel,))

deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

for i in range(0,10):
    N = 400000
    test_label = np.random.randint(M, size=N)
    test_data = []

    for i in test_label:
        temp = np.zeros(M)
        temp[i] = 1
        test_data.append(temp)

    test_data = np.array(test_data)

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    encoded_signal = encoder.predict(test_data)
    stddev = np.std(encoded_signal)
    print('std', stddev)
    mean = np.mean(encoded_signal)
    print('mean', mean)



