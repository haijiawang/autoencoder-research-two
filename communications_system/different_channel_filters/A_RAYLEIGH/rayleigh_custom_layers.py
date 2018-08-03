from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import sqrt
import random


ch_matrix = [1.17218346, 0.0905365, 1.09069497]

ch_matrix_slow=[[0.32148723, 0.48041021, 1.33720521],
[0.9199546, 0.48261801, 0.51533739],
[0.26904993,1.24381704, 0.87606228],
[1.58291486, 0.41295955, 1.17637462],
[0.75668606, 0.18396639, 1.81039747],
[2.2151082,  0.3094984,  0.67245506],
[0.12456818, 0.3047424, 1.62468057],
[0.55540213, 1.33643236, 1.71097225],
[1.03043178, 0.60816349, 0.65257436],
[1.50544855, 0.45524064, 0.4651089 ]]

'''
INITIALIZED
'''
n_channel = 3
samp_size = 20

ch_matrix = np.random.seed(0)
ch_matrix= (np.sqrt((np.random.randn(3) ** 2) + (np.random.randn(3) ** 2)) / np.sqrt(2.0))
print(ch_matrix)

ch_matrix_slow = np.random.seed(1)
ch_matrix_slow = (np.sqrt((np.random.randn(samp_size,n_channel) ** 2) + (np.random.randn(samp_size,n_channel) ** 2)) / np.sqrt(2.0))
print(ch_matrix_slow)

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


class RayleighSlow(Layer):


    def __init__(self, stddev, i , **kwargs):
        super(RayleighSlow, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.i = i

    def call(self, inputs, training=None):
        def noised():
            #ch_matrix = sqrt((np.random.randn(3) ** 2)+ (np.random.randn(3) ** 2)) / sqrt(2.0)
            #ch_coeff = sqrt(random.gauss(0, 1) ** 2.0 + random.gauss(0, 1) ** 2.0) / sqrt(2.0)
            print(ch_matrix_slow[self.i])
            return (inputs * ch_matrix_slow[self.i]) + K.random_normal(shape=K.shape(inputs),
                                            mean=0,
                                            stddev=self.stddev)

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        config = {'i': self.i}
        base_config = super(RayleighSlow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

