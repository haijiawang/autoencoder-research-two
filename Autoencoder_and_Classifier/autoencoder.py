from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

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

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))



import matplotlib as mpl
mpl.use('TkAgg')

#use Matplotlib
import matplotlib.pyplot as plt

autoencoder_train = autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))

epochs=10
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
alpha = 1
plt.figure()
#plt.plot(epochs, (loss * alpha), 'bo', label='Training loss')
plt.plot(epochs, (val_loss * alpha), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#visualize reconstructed inputs and the encoded representations
#encoded and decode some digits
#note that we take them from the TEST set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10 #how many digits we will display
plt.figure(figsize=(20, 4))

for i in range(n):
    #display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
