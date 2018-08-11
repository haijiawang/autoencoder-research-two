from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import sqrt
import random
n_channels = 3
samp_size = 10
Kz = 4

X_matrix = np.random.seed(2)
X_matrix = np.random.randn()

Y_matrix = np.random.seed(3)
Y_matrix = np.random.randn()

X_slow_matrix = np.random.seed(4)
X_slow_matrix = np.random.randn(samp_size)

Y_slow_matrix = np.random.seed(5)
Y_slow_matrix = np.random.randn(samp_size)

class Rician(Layer):

    def __init__(self, stddev, **kwargs):
        super(Rician, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            mu = np.sqrt(Kz) / np.sqrt(Kz + 1)
            sigma = 1 / (np.sqrt(2 * (Kz + 1)))
            X = (sigma * X_matrix) + mu
            Y = (sigma * Y_matrix)
            Z = np.sqrt((X ** 2) + (Y ** 2))

            return (Z * inputs) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(Rician, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class RicianSlow(Layer):

    def __init__(self, stddev, i,  **kwargs):
        super(RicianSlow, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.i = i

    def call(self, inputs, training=None):
        def noised():
            mu = np.sqrt(Kz) / np.sqrt(Kz + 1)
            sigma = 1 / (np.sqrt(2 * (Kz + 1)))
            X = (sigma * X_slow_matrix[self.i]) + mu
            Y = (sigma * Y_slow_matrix[self.i])
            Z = np.sqrt((X ** 2) + (Y ** 2))

            return (Z * inputs) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        config = {'i': self.i}
        base_config = super(RicianSlow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class CustomNormalization(Layer):
    def __init__(self, stddev, **kwargs):
        super(CustomNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev


    def call(self, inputs, training=None):
        def normalized():
            initial_m = K.mean(inputs)
            initial_std = K.std(inputs)
            m = 0.0
            std = 1.0
            var = 1.0
            epsilon = 0.001
            gamma = 12.0
            beta = 1.0
            return (((inputs - (m)) / (np.sqrt(var + epsilon)) * gamma)) + 0.0


        return K.in_train_phase(normalized, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(CustomNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

