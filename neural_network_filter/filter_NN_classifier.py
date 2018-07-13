from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras import regularizers
import matplotlib
matplotlib.use('agg')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from skimage import exposure
import pylab
from skimage import io, color
from scipy.misc import imread
from PIL import ImageFilter
from PIL import Image
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


'''
SPLITTING THE DATA INTO DIFFERENT CATEGORIES
'''
X_test = np.copy(x_test)
X_train = np.copy(x_train)

'''
PREPARING THE "BLURRY" IMAGES (AUTOENCODER OUTPUT)
'''

epochs_a = 5

encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation = 'sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['accuracy'])

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))

autoencoder_train = autoencoder.fit(X_train, X_train,
                epochs=epochs_a,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(X_test, X_test))


encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)


'''
PREPARING THE "SHARP" IMAGES 
'''
#The SHARP images are x_sharp

sharp_img = np.copy(x_test)
x_sharp = np.copy(x_test)
for i in range(10000):
    img = Image.fromarray(sharp_img[i])
    img = img.filter(ImageFilter.SHARPEN)
    #plt.imshow(img)
    x_sharp[i] = np.array(img)


'''
BUILDING THE NEURAL NETWORK
'''
#specifying the input dimensions
input_img = Input(shape=(28, 28, 1))
x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2,2), padding='same')(x)
#x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)
#at this point the representation is (4, 4, 8) i.e. 128-dimensional
#at this point the representation is (7, 7, 32)
x = Conv2D(64, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
#x = Conv2D(16, (3,3), activation='relu')(x)
#x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

conv_nn = Model(input_img, decoded)
conv_nn.compile(optimizer='adam', loss='binary_crossentropy')

#PREPARE THE DATA
x_sharp = x_sharp.astype('float32') / 255
x_sharp = np.reshape(x_sharp, (len(x_sharp), 28, 28, 1))
decoded_imgs = np.reshape(decoded_imgs, (len(decoded_imgs), 28, 28, 1))

x_sharp_train = x_sharp[:6000]
x_sharp_test = x_sharp[:4000:]
decoded_imgs_train = decoded_imgs[:6000]
decoded_imgs_test = decoded_imgs[:4000:]
from keras.callbacks import TensorBoard


conv_nn.fit(decoded_imgs_train, x_sharp_train,
            epochs=2,
            batch_size=128,
            shuffle=True,
            validation_data = (decoded_imgs_test, x_sharp_test),
            callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

#RESULTS
new_decoded_imgs = conv_nn.predict(decoded_imgs_test)


n=10
plt.figure(figsize=(20,6))
for i in range(n):
    #display ORIGINAL
    ax = plt.subplot(3, n, i +1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    plt.title('Autoencoder Input', fontsize=4)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display autoencoder produced
    ax = plt.subplot(3, n, i+1 +n)
    plt.imshow(decoded_imgs_test[i].reshape(28,28))
    plt.gray()
    plt.title('Autoencoder Output', fontsize=4)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(3, n, i+1+ n +n)
    plt.imshow(new_decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.title('Sharpened', fontsize=6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



'''
CLASSIFIER PORTION
'''

import os
os.environ['CUDA_VISIBLE_DEVICES']=''

class_x = np.copy(x_train)
class_x = class_x.astype('float32')/255
class_x = class_x.reshape((len(class_x)), np.prod(class_x.shape[1:]))
new_decoded_imgs = new_decoded_imgs.reshape(4000, 784)

n_classes = 10
y_test = y_test[:4000:]
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#training the model and saving metrics in history
history = model.fit(class_x, y_train,
                    batch_size=128, epochs=5,
                    verbose=2,
                    validation_data=(new_decoded_imgs, y_test))

#saving the model
save_dir = "/Users/alyssa/Documents/keras-practice-two/neural_network_filter/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


mnist_model = load_model("keras_mnist.h5")
loss_and_metrics = mnist_model.evaluate(new_decoded_imgs, y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

#load the model and create predictions on the test set
mnist_model = load_model("keras_mnist.h5")
predicted_classes = mnist_model.predict_classes(new_decoded_imgs)

#see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print(len(correct_indices), "classified correctly")
print(len(incorrect_indices), "classified incorrectly")

#adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7, 14)

plt.figure()

#plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(new_decoded_imgs[correct].reshape(28,28), cmap='gray',
               interpolation='none')
    plt.title("Predicted: {}, Truth: {}".format(predicted_classes[correct], y_test[correct]), fontsize=5)
    plt.xticks([])
    plt.yticks([])

#plot 9 incorrect
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+15)
    plt.imshow(new_decoded_imgs[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(predicted_classes[incorrect], y_test[incorrect]), fontsize=5)
    plt.xticks([])
    plt.yticks([])


plt.show()