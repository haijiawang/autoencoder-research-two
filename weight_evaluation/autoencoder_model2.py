from keras.layers import Input, Dense, Layer
from keras.models import Model
from keras import regularizers
from numpy import linalg as LA
from keras.layers.core import Activation
from keras.utils import np_utils
#this is the size of our encoded representations
encoding_dim = 2 #32 floats--> compression of factor 24.5, assuming the input is 784 floats

#this is our input placeholder
input_img = Input(shape=(784,))

#DENSE LAYERS

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation = 'sigmoid')(decoded)

#this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

#encoder model
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(encoding_dim,))
#encoded_input = Input(shape=(encoding_dim,))

deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test,_) = mnist.load_data()

#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))
print x_train.shape
print x_test.shape

'''autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))
'''


import matplotlib as mpl
mpl.use('TkAgg')

#use Matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

autoencoder_model = autoencoder.fit(x_train, x_train,
                epochs=75,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))

last_weight_matrix = autoencoder.layers[-4].get_weights()[0]
last_weight_matrix = last_weight_matrix.reshape(64, encoding_dim)
last_weight_matrix = last_weight_matrix.astype('float')
last_weight_matrix = preprocessing.scale(last_weight_matrix)
u, s, vh = np.linalg.svd(last_weight_matrix)
plt.figure()
plt.plot(s)
plt.title('Plot of S Values for Weight Matrix')
plt.show()
