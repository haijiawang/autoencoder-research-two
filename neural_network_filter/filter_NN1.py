from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
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

'''
SPLITTING THE DATA INTO DIFFERENT CATEGORIES
'''
X_test = np.copy(x_test)
X_train = np.copy(x_train)

'''
PREPARING THE "BLURRY" IMAGES (AUTOENCODER OUTPUT)
'''

epochs_a = 10

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
conv_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

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
            epochs=500,
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
    plt.title('Autoencoder Input', fontsize=6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display autoencoder produced
    ax = plt.subplot(3, n, i+1 +n)
    plt.imshow(decoded_imgs_test[i].reshape(28,28))
    plt.gray()
    plt.title('Autoencoder Output', fontsize=6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(3, n, i+1+ n +n)
    plt.imshow(new_decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.title('Sharpened', fontsize=9)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()




