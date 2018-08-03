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
import itertools

# defining parameters
M = 128
k = np.log2(M)
k = int(k)
print ('M:', M, 'k:', k)


# generating data of size N
#N = 10000
temp = list(itertools.product([0,1], repeat=k))
for i in temp:
    i = np.asarray(i)
#temp = np.asarray(temp)
t=[]
for i in range(0, 1500):
    for j in temp:
        t.append(j)

data= np.asarray(t)

R = 7.0 / 7.0
n_channel = 7
input_signal = Input(shape=(k,))
encoded = Dense(k, activation='relu')(input_signal)
encodedT = Dense(k, activation='relu')(encoded)
encodedTT = Dense(k, activation='relu')(encodedT)
encoded1 = Dense(n_channel, activation='linear')(encodedTT)
encoded2 = BatchNormalization()(encoded1)

n = np.random.randint(-4,8)
EbNo_train = np.power(10, n/10.0)  # coverted 7 db of EbNo
# EbNo_train = EbNo_train.astype('float')
alpha1 = pow((2 * R * EbNo_train), -0.5)
encoded3 = GaussianNoise(alpha1)(encoded2)

decoded = Dense(k, activation='relu')(encoded3)
decodedT = Dense(k, activation='relu')(decoded)
decodedTT = Dense(2*k, activation='relu')(decodedT)
decoded1 = Dense(k, activation='sigmoid')(decodedTT)

autoencoder = Model(input_signal, decoded1)
# sgd = SGD(lr=0.001)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


val = list(itertools.product([0,1], repeat=k))
for i in val:
    i = np.asarray(i)
#val = np.asarray(val)
v=[]
for i in range(0, 500):
    for j in val:
        v.append(j)

val_data = np.asarray(v)

autoencoder.fit(data, data,
                epochs=150,
                batch_size=300,
                verbose=2,
                validation_data=(val_data, val_data))

from keras.models import load_model

encoder = Model(input_signal, encoded2)

encoded_input = Input(shape=(n_channel,))

deco = autoencoder.layers[-4](encoded_input)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

#N = 400000
#test_label = np.random.randint(M, size=N)
#test_data = []

test = list(itertools.product([0,1], repeat=k))
for i in test:
    i = np.asarray(i)
#test = np.asarray(test)
z= []
for i in range(0, 15000):
    for j in test:
        z.append(j)
test_data = np.asarray(z)

N = test_data.shape[0]
print(N)
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


EbNodB_range = list(frange(-4, 8.5, 0.5))
ber = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
    # print('Test' , EbNo)
    # print('R value', R)
    alpha1 = (2 * R * EbNo) ** (-0.5)
    noise_std = alpha1
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn, k)
    encoded_signal = encoder.predict(test_data)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder.predict(final_signal)
    #pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_final_signal != test_data)
    no_errors = no_errors.astype(int)
    no_errors = no_errors.sum()
    ber[n] = 1.0 * no_errors / nn
    print ('SNR:', EbNodB_range[n], 'BER:', ber[n])

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

'''
PLOTTING THE AUTOENCODER
'''
plt.plot(EbNodB_range, ber, linestyle='--', color='b', marker='o', label='Autoencoder(7,7)')

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

plt.savefig('binary_77')
plt.show()