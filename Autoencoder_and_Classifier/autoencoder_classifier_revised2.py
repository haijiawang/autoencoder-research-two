from keras.layers import Input, Layer
from keras.models import Model
from keras import regularizers
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


epochs_a = 100
'''
#AUTOENCODER
'''
#this is the size of our encoded representations
encoding_dim = 32 #32 floats--> compression of factor 24.5, assuming the input is 784 floats

#this is our input placeholder
input_img = Input(shape=(784,))

#DENSE LAYERS
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation = 'sigmoid')(decoded)

#Classifier LAYERS
#classified = Dense(784, activation='relu')(decoded)
classified = Dense(512, activation='relu')(decoded)
#classified = Dropout(0.2)(classified)
classified = Dense(512, activation='relu')(classified)
#classified = Dropout(0.2)(classified)
classified = Dense(10, activation='softmax')(classified)

#this model maps an input to its reconstruction
model = Model(input_img, classified)

#encoder model
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
deco = model.layers[-6](encoded_input)
deco = model.layers[-5](deco)
deco = model.layers[-4](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

decoded_input = Input(shape=(784,))
#classi = model.layers[-4](decoded_input)
classi = model.layers[-3](decoded_input)
classi = model.layers[-2](classi)
classi = model.layers[-1](classi)
classifier = Model(decoded_input, classi)

#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
PREPARING THE DATA
'''
#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

auto_class_train = model.fit(X_train, Y_train,
                epochs=2,
                batch_size=128,
                verbose=2,
                validation_data=(X_test, Y_test))


encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)
classified_imgs = classifier.predict(decoded_imgs)

final_results = model.predict(X_test)

n = 10 #how many digits we will display
#plt.figure(figsize=(20, 4))


correct_indices = np.nonzero(final_results == y_test)[0]
incorrect_indices = np.nonzero(final_results != y_test)[0]



for i in range(n):
    #display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, 1+i+n+n)
    plt.gray()
    plt.title(np.argmax(final_results[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

'''
plt.figure()
#plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray',
               interpolation='none')
    plt.title("Predicted: {}, Truth: {}".format(final_results[correct], y_test[correct]), fontsize = 9)
    plt.xticks([])
    plt.yticks([])

#plot 9 incorrect
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(final_results[incorrect], y_test[incorrect]), fontsize = 9)
    plt.xticks([])
    plt.yticks([])

#figure_evaluation.show()
plt.show()
'''