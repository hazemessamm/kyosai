from __future__ import absolute_import
import set_path
from initializers import initializers
from activations import activations
from jax import numpy as jnp
from jax.experimental import stax
from functools import wraps
from jax.random import PRNGKey
import construction_layers as cl

class Input:
    def __init__(self, shape=None):
        if not shape or not isinstance(shape, tuple):
            raise layer_exceptions.InputShapeNotFoundException(f'shape should have value in a tuple, found {shape}')
        self.shape = shape
        self.batch_size = None

    def get_shape(self):
        return self.shape

    def set_batch_size(self, batch_size):
        if batch_size > 0:
            self.batch_size = batch_size
        else:
            raise Exception('Batch size should be bigger than zero')

    def store_data(self, x, y):
        self.training_data = X
        self.training_labels = y
        self.stored_data = True

    def __call__(self, index):
        current_batch_index = index*self.batch_size
        current_batch_x = self.training_data[current_batch_index: current_batch_index+self.batch_size]
        current_batch_y = self.training_labels[current_batch_index: current_batch_index+self.batch_size]

        return current_batch_x, current_batch_y

class Layer:
    def __init__(self):
        self.params = ()
        self.prev = None
        self.built=False

    def get_initializer(self, identifier):
        return initializers.get(identifier)

    def get_activation(self, identifier):
        return activations.get(identifier)

    def connect(self, layer):
        self.prev = layer
    
    def get_weights(self):
        return self.params

    def set_weights(self, new_weights):
        if not isinstance(new_weights, tuple):
            raise Exception(f"Weights should be inside a tuple example: (W, b), found {type(new_weights)}")

        for current_p, new_p in zip(self.params, new_weights):
            if current_p.shape != new_p.shape:
                raise Exception(f"New weights is not compatible with the current weight shapes, {current_p.shape} != {new_p.shape}")
            else:
                self.params += (new_weights,)


class Dense(Layer):
    def __init__(self, units, activation=None, kernel_initializer='glorot_normal', bias_initializer='normal', 
    input_shape=None, key=PRNGKey(1)):
        super(Dense, self).__init__()
        self.units = units
        self.activation = activation
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.key = key
        self.init_fn, self.apply_fn = stax.Dense(units, W_init=self.kernel_initializer, b_init=self.bias_initializer)
    
    def get_kernel_shape(self):
        return self.kernel_shape

    def get_bias_shape(self):
        return self.bias_shape

    def get_output_shape(self):
        return self.shape

    def get_input_shape(self):
        return self.input_shape

    def build(self, input_shape):
        self.shape, self.params = self.init_fn(rng=self.key, input_shape=input_shape)
        self.input_shape = input_shape
        self.kernel_shape = self.params[0].shape
        self.bias_shape = self.params[0].shape
        self.built = True

    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=params)
        return self.activation(out) if self.activation is not None else out

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            if isinstance(inputs, Layer) or isinstance(inputs, Input):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            else:
                self.build(inputs.shape)
                if self.input_shape != inputs.shape:
                    raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
                else:
                    out = self.apply_fn(inputs=inputs, params=self.params)
                    return self.activation(out) if self.activation is not None else out

    def __repr__(self):
        if self.built:
            return f"Dense Layer with input shape {self.input_shape} and output shape {self.shape}"
        else:
            return "Dense Layer"



class Flatten(Layer):
    def __init__(self, key=PRNGKey(1)):
        super(Flatten, self).__init__()
        self.init_fn, self.apply_fn = stax.Flatten
        self.key = key

    def build(self, input_shape):
        self.shape, self.params = self.init_fn(input_shape=input_shape, rng=self.key)
        self.input_shape = input_shape
        self.built = True

    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=params)
        return out

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            if isinstance(inputs, Layer) or isinstance(inputs, Input):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            else:
                self.build(inputs.shape)
                if self.input_shape != inputs.shape:
                    raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
                else:
                    out = self.apply_fn(inputs=inputs, params=self.params)
                    return out

    def __repr__(self):
        if self.built:
            return f"Flatten Layer with input shape {self.input_shape} and output shape {self.shape}"
        else:
            return "Flatten Layer"

class Dropout(Layer):
    def __init__(self, rate, training=False, key=PRNGKey(1)):
        self.rate = rate
        if training:
            self.mode = 'train'
        else:
            self.mode = 'predict'
        self.key = key
        self.init_fn, self.apply_fn = stax.Dropout(rate=rate, mode=mode)
    
    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=params)
        return out
    
    def build(self, input_shape):
        output_shape, self.params = self.init_fn(rng=self.key, input_shape=input_shape)
        self.input_shape = input_shape
        self.kernel_shape = params[0].shape
        self.bias_shape = params[0].shape
        self.built = True
        self.shape = output_shape

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            if isinstance(inputs, Layer) or isinstance(inputs, Input):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            else:
                self.build(inputs.shape)
                if self.input_shape != inputs.shape:
                    raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
                else:
                    out = self.apply_fn(inputs=inputs, params=self.params)
                    return out


class Activation(Layer):
    def __init__(self, identifier):
        super(Activation, self).__init__()
        self.identifier = identifier
        self.init_fn, self.apply_fn = cl.Activation(identifier)
    
    def build(self, input_shape):
        self.input_shape, self.params = self.init_fn(input_shape)
        self.shape = self.input_shape
        self.built=True


    def call_with_external_weights(self, inputs, params):
        out = self.activation(inputs)
        return out

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            if isinstance(inputs, Layer) or isinstance(inputs, Input):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            else:
                self.build(inputs.shape)
                if self.input_shape != inputs.shape:
                    raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
                else:
                    out = self.activation(inputs)
                    return out

        

    
