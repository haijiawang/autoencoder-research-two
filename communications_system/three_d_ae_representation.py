# importing libs# import
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import random as rn
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

N = 1500
test_label = np.random.randint(M, size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)

test_data = np.array(test_data)

temp_test = 6
print (test_data[temp_test][test_label[temp_test]], test_label[temp_test])
print(encoder.predict(val_data).shape)
print(encoder.predict(val_data)[0])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_emb = encoder.predict(test_data)
noise_std = np.sqrt(1/(2*R*EbNo_train))
noise = noise_std * np.random.randn(N, n_channel)
x_emb = x_emb + noise
#print (x_emb[0])
x_x_emb = x_emb[:,0]
np.save('x_axis', x_x_emb)
#print(x_x_emb[0])
y_x_emb = x_emb[:,1]
np.save('y_axis', y_x_emb)

z_x_emb = x_emb[:,2]
np.save('z_axis', z_x_emb)
np.save('corrdinates', x_emb)

print(y_x_emb[0])
print(z_x_emb[0])
ax.scatter(x_x_emb, y_x_emb, z_x_emb)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

'''
from sklearn.manifold import TSNE
X_embedded = TSNE(learning_rate=600, n_components=2, n_iter=500, random_state=0, perplexity=30).fit_transform(x_emb)
print('X_embedded shape:', X_embedded.shape)
X_embedded = X_embedded
X_new_embedded = X_embedded / 5
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.grid()
plt.title('(5,5) Constellation Signal')
plt.savefig('Constellation_5_5')
plt.show()

'''

