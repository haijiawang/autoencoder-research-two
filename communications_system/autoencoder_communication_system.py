# -*- coding: utf-8 -*-
"""
Edited Version - Haijia

"""

# importing libs# import
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import random as rn

# defining parameters
M = 4
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

R = 2.0 / 2.0
n_channel = 2
input_signal = Input(shape=(M,))
encoded = Dense(M, activation='relu')(input_signal)
encoded1 = Dense(n_channel, activation='linear')(encoded)
encoded2 = BatchNormalization()(encoded1)

EbNo_train = np.power(10, 0.7)  # coverted 7 db of EbNo
#EbNo_train = EbNo_train.astype('float')
alpha1 = pow((2 * R * EbNo_train), -0.5)
encoded3 = GaussianNoise(alpha1)(encoded2)

decoded = Dense(M, activation='relu')(encoded3)
decoded1 = Dense(M, activation='softmax')(decoded)

autoencoder = Model(input_signal, decoded1)
# sgd = SGD(lr=0.001)
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
                epochs=20,
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

N = 400000
test_label = np.random.randint(M, size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)

test_data = np.array(test_data)

temp_test = 6
print (test_data[temp_test][test_label[temp_test]], test_label[temp_test])

print('val data', val_data)
predicted = np.argmax(autoencoder.predict(val_data), axis =1)
print('predicted', predicted)

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


EbNodB_range = list(frange(-4, 8.5, 0.5))
ber = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
    #print('Test' , EbNo)
    #print('R value', R)
    alpha1 = (2 * R * EbNo) ** (-0.5)
    noise_std = alpha1
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn, n_channel)
    encoded_signal = encoder.predict(test_data)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int)
    no_errors = no_errors.sum()
    ber[n] = 1.0 * no_errors / nn
    print ('SNR:', EbNodB_range[n], 'BER:', ber[n])

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(EbNodB_range, ber, linestyle='--', color='b', marker='o', label='Autoencoder(11,11)')
print(ber)
np.save('auto_11_11', ber)
# plt.plot(EbNodB_range, ber, linestyle='', marker='o', color='r')
# plt.plot(EbNodB_range, ber, linestyle='-', color = 'b')
# plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('AutoEncoder_11_11_BER_matplotlib')
plt.show()