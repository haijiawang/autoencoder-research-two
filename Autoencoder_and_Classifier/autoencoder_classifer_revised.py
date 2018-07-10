from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np

alpha = 1
beta = 1

epochs_a = 100
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

#this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

#encoder model
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


encoded_input = Input(shape=(encoding_dim,))

deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['accuracy'])


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))
print X_train.shape
print X_test.shape


autoencoder_train = autoencoder.fit(X_train, X_train,
                epochs=epochs_a,
                batch_size=256,
                shuffle=True,
                verbose=2,
                validation_data=(X_test, X_test))


encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

#CLASSIFIER


import os
os.environ['CUDA_VISIBLE_DEVICES']=''

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


#PREPARING CLASSIFER INPUT DATA: input is decoded images from autoencoder
if beta != 0:
    epochs_b = 100

    '''
    If alpha is 0, then SET classifier_input = X_test
    If alpha is nonzero, then SET classifier_input = decoded_imgs
    '''
    classifier_input = decoded_imgs

    #one-hot encoding using keras' numpy-related utilities (1 in the position of the value)
    n_classes = 10
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)


    #building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    #compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
                        batch_size=128, epochs= epochs_b,
                        verbose=2,
                        validation_data=(classifier_input, Y_test))

    #saving the model
    save_dir = "/Users/alyssa/Documents/keras-practice-two/Autoencoder_and_Classifier/"
    model_name = 'keras_mnist.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


    plt.figure()
    plt.plot(history.history['loss'], label ='Loss for Classifier')
    plt.plot(history.history['val_loss'], label='Validation Data Loss for Classifier')
    plt.plot(autoencoder_train.history['loss'], label='Loss for Autoencoder')
    plt.plot(autoencoder_train.history['val_loss'], label='Validation Loss for Autoencoder')
    plt.title('Epoch vs. Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show(block=False)


    mnist_model = load_model("keras_mnist.h5")
    loss_and_metrics = mnist_model.evaluate(classifier_input, Y_test, verbose=2)

    print("Test Loss", loss_and_metrics[0])
    #print("Test Accuracy", loss_and_metrics[1])

    #load the model and create predictions on the test set
    mnist_model = load_model("keras_mnist.h5")
    predicted_classes = mnist_model.predict_classes(classifier_input)

    #see which we predicted correctly and which not
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    print()
    print(len(correct_indices), "classified correctly")
    print(len(incorrect_indices), "classified incorrectly")

    #adapt figure size to accomodate 18 subplots
    plt.rcParams['figure.figsize'] = (7, 14)
    if alpha == 0:
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

    if (alpha and beta != 0):
        n = 10
        plt.figure()


        #plt.figure(figsize=(20, 4))

        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(X_test[i].reshape(28, 28))
            plt.title("X")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.title("X'\nP: {},\n T: {}".format(predicted_classes[i], y_test[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.set_xlabel("P: {},\n T: {}".format(predicted_classes[i], y_test[i]))


        plt.show()



