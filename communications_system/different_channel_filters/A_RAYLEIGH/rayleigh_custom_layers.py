from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import sqrt
import random
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.legacy import interfaces
from keras.engine.base_layer import Layer, InputSpec

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


class BatchNormalization1(Layer):

    def __init__(self,
                 axis=1,
                 momentum=0.001,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalization1, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalization1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class CustomNormalizationX(Layer):
    def __init__(self, stddev, **kwargs):
        super(CustomNormalizationX, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev


    def call(self, inputs, training=None):
        def normalized():
            initial_m = K.mean(inputs)
            initial_std = K.std(inputs)
            m = 0.0
            std = 1.0

            return ((inputs - (initial_m)) * (( self.stddev / initial_std)))


        return K.in_train_phase(normalized, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(CustomNormalizationX, self).get_config()
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
            gamma = 8.0
            beta = 1.0
            return (((inputs - (m)) / (np.sqrt(var + epsilon)) * gamma)) + 0.0


        return K.in_train_phase(normalized, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(CustomNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape