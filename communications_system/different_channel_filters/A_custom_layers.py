from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import sqrt
import random

class GaussianNoise1(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoise1, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


ch_matrix = [1.17218346, 0.0905365, 1.09069497]



class Rayleigh(Layer):


    def __init__(self, stddev, **kwargs):
        super(Rayleigh, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            #ch_matrix = sqrt((np.random.randn(3) ** 2)+ (np.random.randn(3) ** 2)) / sqrt(2.0)
            #ch_coeff = sqrt(random.gauss(0, 1) ** 2.0 + random.gauss(0, 1) ** 2.0) / sqrt(2.0)
            return (inputs * ch_matrix) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(Rayleigh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



class Rician(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, Kz, **kwargs):
        super(Rician, self).__init__(**kwargs)
        self.supports_masking = True
        self.Kz = Kz

    def call(self, inputs, training=None):
        def riciand():
            self.__rician_mu = sqrt(self.Kz / (self.Kz + 1))
            self.__rician_s = sqrt(1 / (2 * (self.Kz + 1)))
            self.__rician_chn = self.__rician_s * (random.normalvariate(0,1) + random.normalvariate(0,1) * 1j) + self.__rician_mu
            return self.__rician_chn * inputs

        return K.in_train_phase(riciand, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(Rician, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Rician1(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(Rician1, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            Kz = 4
            mu = np.sqrt(Kz) / np.sqrt(Kz + 1)
            sigma = 1 / (np.sqrt(2 * (Kz + 1)))
            X = (sigma * np.random.randn()) + mu
            Y = (sigma * np.random.randn())
            Z = np.sqrt((X ** 2) + (Y ** 2))

            return (Z * inputs) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(Rician1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Rayleigh1(Layer):
    def __init__(self, stddev, **kwargs):
        self.stddev = stddev
        self.supports_masking = True
        super(Rayleigh1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape,),
                                      initializer='uniform',
                                      trainable=True)
        super(Rayleigh1, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        def noised():
            ch_coeff = sqrt(random.gauss(0, 1) ** 2.0 + random.gauss(0, 1) ** 2.0) / sqrt(2.0)
            return (inputs * ch_coeff) + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)

        return K.in_train_phase(noised, inputs)


    def compute_output_shape(self, input_shape):
        return input_shape