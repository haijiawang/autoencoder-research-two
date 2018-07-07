from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_dim = 32

#this is our input placeholder
input_img = Input(shape=(392,))


encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(392, activation = 'sigmoid')(decoded)


#this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

#encoder model
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(encoding_dim,))
encoded_input = Input(shape=(encoding_dim,))

deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)

# create the decoder model
decoder = Model(encoded_input, deco)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(train, _),(test, _) = mnist.load_data()
#train = train.reshape(60000, 784)
#test = test.reshape(10000, 784)

train_left = train[:, :, :14]
train_right = train[:, :, 14:]
test_left = test[:, :, :14]
test_right = test[:, :, 14:]
#y_train_right = y_train_right[:, :, 14]
#x_test_left = x_test_left[:, :, :14]
#y_test_right = y_test[:, :, 14]

#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
train_left = train_left.astype('float32')/255
test_left = test_left.astype('float32') / 255
train_right = train_right.astype('float32')/255
test_right = test_right.astype('float32') / 255
train_left = train_left.reshape((len(train_left)), np.prod(train_left.shape[1:]))
test_left = test_left.reshape((len(test_left)), np.prod(test_left.shape[1:]))
train_right = train_right.reshape((len(train_right)), np.prod(train_right.shape[1:]))
test_right = test_right.reshape((len(test_right)), np.prod(test_right.shape[1:]))
print train_left.shape
print test_right.shape


#xtrain-left: xtrain-right
autoencoder.fit(train_left, train_right,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(test_left, test_right))

#visualize reconstructed inputs and the encoded representations
#encoded and decode some digits
#note that we take them from the TEST set
encoded_imgs = encoder.predict(test_left)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib as mpl
mpl.use('TkAgg')

#use Matplotlib
import matplotlib.pyplot as plt

n = 10 #how many digits we will display
plt.figure(figsize=(20, 4))

for i in range(n):
    #display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test_left[i].reshape(28, 14))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 14))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
