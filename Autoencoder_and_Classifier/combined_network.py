from keras.layers import Input, Dense
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


(X_train, y_train), (X_test, y_test) = mnist.load_data()



epochs_a = 10
#
#AUTOENCODER
#
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

classified = Dense(512, activation='relu')(decoded)
classified= Dense(512, activation='relu')(classified)
classified = Dense(10, activation='softmax')(classified)

#this model maps an input to its reconstruction
autoencoder = Model(input_img, classified)

#encoder model
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
deco = autoencoder.layers[-6](encoded_input)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

decoded_input = Input(shape=(784,))
clas = autoencoder.layers[-3](decoded_input)
clas = autoencoder.layers[-2](clas)
clas = autoencoder.layers[-1](clas)
classifier = Model(decoded_input, clas)

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])



#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))
print X_train.shape
print X_test.shape


autoencoder_train = autoencoder.fit(X_train, Y_train,
                epochs=epochs_a,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(X_test, Y_test))


plt.figure()
plt.plot(autoencoder_train.history['loss'], 'b', label='loss')
plt.plot(autoencoder_train.history['val_loss'], label = 'validation loss')
plt.plot(autoencoder_train.history['acc'], label = 'accuracy')
plt.plot(autoencoder_train.history['val_acc'], label = 'validation accuracy')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#visualize reconstructed inputs and the encoded representations
#encoded and decode some digits
#note that we take them from the TEST set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)


predicted_classes = autoencoder_train.predict_classes(y_test)

n = 10
for i in n:
    plt.figure()
    plt.subplot(2, 5, i)
    plt.title("Predicted: {}, Truth: {}".format(predicted_classes[i], y_test[i]))
    plt.xticks([])
    plt.yticks([])


#see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]


#adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7, 14)

plt.figure()
plt.rcParams.update({'font.size': 5})
# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6, 3, i + 1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray',
               interpolation='none')
    plt.title("Predicted: {}, Truth: {}".format(predicted_classes[correct], y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6, 3, i + 10)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.show()





