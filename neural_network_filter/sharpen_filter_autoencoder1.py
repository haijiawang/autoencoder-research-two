from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test,_) = mnist.load_data()
from skimage import exposure
import pylab
from skimage import io, color
from PIL import ImageFilter
from PIL import Image

'''
AUTOENCODER SECTION
'''
encoding_dim = 32

input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation = 'sigmoid')(decoded)

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



#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

'''
SHARPENING FILTER SECTION
'''

filter_imgs = np.copy(decoded_imgs) * 255
filter_imgs = filter_imgs.astype('float32')
filter_imgs = filter_imgs.reshape(10000, 28, 28)


#This function takes an iamge and a kernel and returns the convolution of them
    #Takes an image and kernel
    #Returns a numpy array of size image height by image width (convolution output)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

kernel = np.flipud(np.fliplr(kernel)) #flip the kernel

for i in range(100):
    img_sharpen = np.zeros_like(filter_imgs[i])
    image = filter_imgs[i]
    #image_equalized = exposure.equalize_adapthist(image/np.max(np.abs(image)), clip_limit=0.03)
    #add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] =  image     #puts image onto image with padded zeroes
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            #element-wise multiplication of the kernel and the image
            img_sharpen[y, x] = (kernel*image_padded[y:y+3,x:x+3]).sum()

    filter_imgs[i] = img_sharpen

function_img = np.copy(decoded_imgs) * 255
function_img = function_img.astype('float32')
function_img = function_img.reshape(10000, 28, 28)

for i in range(100):
    function_img[i] = Image.fromarray(function_img[i])
    function_img[i] = function_img[i].filter(ImageFilter.SHARPEN)

#image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)))
#plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)

#img = io.imread('image.jpg')
#img = color.rgb2gray(img)

#adjust the contrast of the image by applying Histogram Equalization
#image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
#plt.imshow(image_equalized, cmap=plt.cm.gray)
#plt.axis('off')
#plt.show()

#convolve the sharpen kernel and the image
#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#image_sharpen = convolve2d(img, kernel)

#plot the filtered image
#plt.imshow(image_sharpen, cmap=plt.cm.gray)
#plt.axis('off')
#lt.show()

#adjust the contrast of the filiter image by applying histogram equalization
#image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)))
#plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
#plt.axis('off')
#plt.show()

n = 10 #how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    #display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display images after going through decoder
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #display images after going through sharpening filter
    ax = plt.subplot(3, n, i+1+n+n)
    plt.imshow(function_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
