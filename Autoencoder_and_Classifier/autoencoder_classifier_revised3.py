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

epochs_a = 10
epochs_b = 10

'''
The main input is going to be our image. 
'''
plt.figure()
    encoding_dim = 32

    #this is our input placeholder
    in1 = Input(shape=(784,))
    #DENSE LAYERS
    encoded = Dense(128, activation='relu')(in1)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    last = Dense(784, activation = 'sigmoid')(decoded)


    in2 = Input(shape=(784,))
    layer = Dense(512, activation='relu', Dropout=0.2)(in2)
    layer = Dense(512, activation='relu', Dropout=0.2)(layer)
    final = Dense(10, activation='softmax')(layer)

    #this model maps an input to its reconstruction
    model = Model(inputs=[in1, in2], outputs=[last, final])


    #encoder model
    #this model maps an input to its encoded representation
    #encoder = Model(in1, encoded)


    #encoded_input = Input(shape=(encoding_dim,))

    #deco = autoencoder.layers[-3](encoded_input)
    #deco = autoencoder.layers[-2](deco)
    #deco = autoencoder.layers[-1](deco)
    # create the decoder model
    #decoder = Model(encoded_input, deco)

    #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    model.compile(optimizer=['adadelta', 'adam'], loss=['binary_crossentropy', 'categorical_crossentropy'], metrics = ['accuracy'])


    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #normalizing all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
    X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))
    print X_train.shape
    print X_test.shape

    model_train = model.fit([X_train, X_train],
                            [X_train, Y_train],
                            batch_size=32,
                            epochs=10, )
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
    save_dir = "/Users/alyssa/Documents/mit_eecs_research/Autoencoder_and_Classifier/"
    model_name = 'keras_mnist.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    mnist_model = load_model("keras_mnist.h5")
    loss_and_metrics = mnist_model.evaluate(classifier_input, Y_test, verbose=2)

    #print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])

    plt.plot(i, loss_and_metrics[1], '-o', color = 'r', label='Classifier Accuracy')

    #load the model and create predictions on the test set
    #mnist_model = load_model("keras_mnist.h5")
    #predicted_classes = mnist_model.predict_classes(classifier_input)

    #see which we predicted correctly and which not
    #orrect_indices = np.nonzero(predicted_classes == y_test)[0]
    #incorrect_indices = np.nonzero(predicted_classes != y_test)[0]


plt.title('k vs. Classifier Accuracy')
plt.show()






