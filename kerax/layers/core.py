from typing import List, Union, Callable
from numpy import ndarray
from sqlalchemy import all_
from kerax.engine import Trackable
from kerax.initializers import initializers
from kerax.initializers import Initializer
from kerax import activations
from jax import numpy as jnp
from jax.example_libraries import stax
from jax.random import PRNGKey
from jax import random
from functools import reduce, cached_property
import operator as op
from jax import lax
from kerax import backend
from jax.numpy import DeviceArray
from kerax.engine.containers import Weight, NodeContainer

class Layer(Trackable):
    def __init__(self, key: PRNGKey = False, trainable: bool = True, dtype: str = 'float32', name: str = None):
        super(Layer, self).__init__(self.__class__.__name__ if name is None else name)
        # Stores the layer params
        self._params = []
        # Stores the previous layer
        self._node_container = NodeContainer()
        self.built = False
        self.key = key
        self.trainable = trainable
        self.output = None
        self.dtype = dtype or backend.precision()

    def in_construction_mode(self, layer):
        return isinstance(layer, Layer)

    def __name__(self):
        return self.name

    @property
    def shape(self):
        return None

    @property
    def input_shape(self):
        if self.built:
            return self._input_shape
        raise Exception(f'Error in {self.name}, Layer is not built yet')

    def compute_output_shape(self, input_shape):
        raise NotImplementedError('Should be implemented in a subclass')

    @property
    def weights(self):
        'returns weights'
        return self._params

    @cached_property
    def params(self):
        return tuple(param.get_weights() for param in self._params)

    @property
    def named_params(self):
        return {param.name: param.get_weights for param in self._params}

    def build(self, input_shape: tuple):
        return NotImplementedError('Should be implemented in a subclass')

    def get_initializer(self, identifier: Union[str, Initializer]):
        'Returns the specified initializer'
        return initializers.get(identifier)

    def get_activation(self, identifier: Union[str, Initializer]):
        'Returns the specified activation'
        return activations.get(identifier)

    def connect(self, layer):
        'Connects the current layer with the previous layer'
        self._node_container.connect_nodes(self, layer)

    def add_weight(self, key: PRNGKey, shape: tuple, initializer: Initializer, dtype: str, name: str, trainable: bool):
        weight = Weight(key, shape, initializer, dtype, name, trainable)
        self._params.append(weight)
        return weight

    def get_weights(self):
        return self._params

    def set_weights(self, new_weights: tuple):
        for w1, w2 in zip(self._params, new_weights):
            w1.set_weights(w2)

    def update_weights(self, new_weights: tuple):
        if self.trainable and len(new_weights) > 0:
            for w1, w2 in zip(self._params, new_weights):
                if w1.trainable:
                    w1.set_weights(w2)

    def __call__(self, inputs, *args, **kwargs):
            if isinstance(inputs, (list, tuple)):
                if not self.built:
                    main_type = type(inputs[0])
                    for i, input_ in enumerate(inputs[1:]):
                        if type(input_) is not main_type:
                            raise Exception(f'Error in layer {self.name}, found input at index {i} is not instance of Layer while the previous are, all of inputs should have the same type')
                        if not hasattr(input_, 'shape'):
                            raise Exception(f'Error in layer {self.name}, found input at index {i} does not have `shape` attribute, all of inputs should have `shape` attribute')

                    if isinstance(inputs[0], Layer):
                        self.build(inputs[0].shape)
                        self.connect(inputs)
                        return self
                    else:
                        self.build(inputs[0].shape)
                        return self.call(inputs, *args, **kwargs)
            elif isinstance(inputs, Layer):
                if self.built:
                    raise Exception(f'Error in layer {self.name}, this layer is already built, you cannot pass any more layers to it')
                else:
                    self.build(inputs.shape)
                    self.connect(inputs)
                    return self
            elif isinstance(inputs, (DeviceArray, ndarray)):
                if self.built:
                    return self.call(inputs, *args, **kwargs)
                else:
                    self.build(inputs.shape)
                    return self.call(inputs, *args, **kwargs)
            else:
                raise Exception(f'Error in layer {self.name}, {type(inputs)} is not supported, supported: (list, tuple, Layer, DeviceArray, ndarray)')

    def call(params, **kwargs):
        raise NotImplementedError('This method should be implemented in Layer subclasses')

    def get_output(self):
        if self.output is None:
            raise Exception('This layer does not have an output yet, call method should be called')
        return self.output

    def __repr__(self):
        if self.built:
            return f'<{self.name} Layer with input shape {self.input_shape} and output shape {self.shape}>'
        else:
            return f'<{self.name} Layer>'

class Input(Layer):
    '''
    Input Layer that stores the training data and returns batches from it 
    params:
    shape: takes a tuple (0, H, W, C)
    '''


    def __init__(self, shape: tuple = None, dtype: str = 'float32', name: str = None):
        super().__init__(key=None, trainable=False, dtype=dtype, name=name)
        if not shape or not isinstance(shape, tuple):
            raise Exception(f'shape should have value in a tuple, found {shape} with type {type(shape)}')
        if shape:
            self._shape, self._input_shape = shape, shape
            self.built = True
        else:
            raise Exception(f'Error in {self.name}, input shape must be provided for the Input Layer')
    
    @property
    def shape(self):
        return self._shape

    @property
    def input_shape(self):
        return self._input_shape

    def build(self, input_shape):
        self._input_shape = input_shape

class Dense(Layer):
    '''

    Dense Layer, (Layer subclass)

    params:
    units: stores number of columns (neurons)
    activation: stores the activation function, default None
    kernel_initializer: stores the kernel initializer, default "glorot_normal"
    bias_initializer: stores the bias initializer, default "normal"

    '''
    def __init__(self, units: int, activation: Union[str, Callable] = None, kernel_initializer: Union[str, Callable] = 'glorot_normal', 
    bias_initializer: Union[str, Callable] = 'normal', use_bias: bool = True, trainable: bool = True, key: PRNGKey = PRNGKey(100), **kwargs):
        super(Dense, self).__init__(key=key, trainable=trainable)
        self.units = units
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.use_bias = use_bias

        input_shape = kwargs.pop('input_shape', False)
        if input_shape:
            self.build(input_shape=input_shape)

    @property
    def shape(self):
        return (1, self.units)

    def compute_kernel_shape(self):
        return self.kernel_weights.shape

    def compute_bias_shape(self):
        return self.bias_weights.shape

    def compute_output_shape(self):
        return (1, self.units)

    def build(self, input_shape: tuple):
        self._input_shape = (input_shape[-1],)
        k1, k2 = random.split(self.key)
        self.kernel_weights = self.add_weight(k1, (input_shape[-1], self.units), self.kernel_initializer, self.dtype, f'{self.name}_kernel', trainable=self.trainable)
        if self.use_bias:
            self.bias_weights = self.add_weight(k2, (self.units,), self.bias_initializer, self.dtype, f'{self.name}_bias', trainable=self.trainable)
        self.built = True

    def dense_op(self, params: tuple, inputs: DeviceArray):
        out = lax.batch_matmul(inputs, params[0])
        if self.use_bias:
            out = jnp.add(out, params[1])
        return out

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output = self.dense_op(params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def call(self, inputs: DeviceArray):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.dense_op(self.params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def __repr__(self):
        if self.built:
            return f"<Dense Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Dense Layer>"

class Flatten(Layer):
    '''
    Flatten Layer, (Layer subclass)
    params: 
    key: Pseudo Random Generator Key, default PRNGKey(1)

    '''
    def __init__(self):
        super(Flatten, self).__init__(key=None, trainable=False)

    @property
    def compute_output_shape(self):
        return (self.input_shape[0], reduce(op.mul, self.input_shape[1:], 1))

    @cached_property
    def shape(self):
        return (reduce(op.mul, self.input_shape[1:], 1),)

    def build(self, input_shape: tuple):
        self._input_shape = input_shape
        self.built = True

    def flatten_op(self, params: tuple, inputs: DeviceArray):
        return lax.reshape(inputs, (inputs.shape[0], *self.shape))

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output =  self.flatten_op(params, inputs)
        return self.output

    def call(self, inputs: DeviceArray, **kwargs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.flatten_op(self.params, inputs)
        return self.output

    def __repr__(self):
        if self.built:
            return f"<Flatten Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Flatten Layer>"

class Dropout(Layer):
    '''
    Dropout Layer, (Layer subclass)
    params:
    rate: probability of turning of a neuron, accepts probability values between 0 and 1
    training: stores the mode of the layer, accepts boolean
    key: Pseudo Random Generator Key, default PRNGKey(1)
    '''
    def __init__(self, rate: float, key: PRNGKey = PRNGKey(1)):
        super(Dropout, self).__init__(key=key)
        self.rate = rate
        self.key = key
    
    @property
    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._input_shape

    def build(self, input_shape: tuple):
        self._input_shape = input_shape
        self.built = True

    def dropout_op(self, params: tuple, inputs: DeviceArray):
        keep = random.bernoulli(self.key, self.rate, inputs.shape)
        return jnp.where(keep, inputs / self.rate, 0)

    def call(self, inputs: DeviceArray, **kwargs):
        'Used during training to pass the parameters while getting the gradients'
        training = kwargs.get('training', False)
        if training:
            self.output = self.dropout_op(self.params, inputs)
        else:
            self.output = inputs
        return self.output
    
    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output = self.dropout_op(params, inputs)
        return self.output

class Activation(Layer):
    '''
    Activation Layer, (Layer subclass)
    params:
    identifier: accepts the activation function as a string or callable
    '''
    def __init__(self, identifier: Union[str, Callable]):
        super(Activation, self).__init__(key=0)
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @property
    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._output_shape
    
    def activation_op(self, params: tuple, inputs: DeviceArray):
        return self.activation(inputs)

    def build(self, input_shape: tuple):
        self._input_shape = input_shape, input_shape
        self.activation = activations.get(self._identifier)
        self.built = True

    def call(self, inputs: DeviceArray, **kwargs):
        self.output = self.activation_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output =  self.activation_op(params, inputs)
        return self.output

    def __repr__(self):
        if self.built:
            return f"<{self._identifier} Activation Layer with input shape {self.input_shape} and output shape {self.shape}>"


class Concatenate(Layer):
    def __init__(self, layers: List[Layer], axis: int = -1, name: str = None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis
        self.layers = layers

    @property
    def shape(self):
        return self._output_shape

    @property
    def output_shape(self):
        return self._input_shape

    def build(self, input_shape: tuple):
        self._input_shape = input_shape
        self.built = True
    
    def concatenate_op(self, params: tuple, inputs: DeviceArray):
        return jnp.concatenate(inputs, axis=self.axis)

    def call(self, inputs: DeviceArray,  **kwargs):
        layer_outputs = tuple(out.get_output() for out in self.layers)
        self.output = self.concatenate_op(self.params, layer_outputs)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output = self.concatenate_op(params, inputs)
        return self.output


class Add(Layer):
    def __init__(self, name: str = None):
        super(Add, self).__init__(name=name)

    @property
    def shape(self):
        return self._output_shape

    @property
    def compute_output_shape(self):
        return self._output_shape

    def add_op(self, params: tuple, inputs: DeviceArray):
        input_1, input_2 = inputs
        return lax.add(input_1, input_2)

    def build(self, input_shape: tuple):
        self._input_shape, self._output_shape = input_shape, input_shape
        self.built = True

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        out = inputs[0]
        for _input in inputs[1:]:
            out = self.add_op(params, (out, _input))
        self.output = out
        return self.output

    def call(self, inputs: DeviceArray):
        output = inputs[0]
        for current_input in inputs[1:]:
            output = self.add_op(self._params, (output, current_input))
        self.output = output
        return self.output
    
    def __repr__(self) -> str:
        if self.built:
            return f"<Add Layer with input shape {self.input_shape} and output shape {self.input_shape}>"
        else:
            return "<Add layer>"

class BatchNormalization(Layer):
    def __init__(self, key: PRNGKey = PRNGKey(100), trainable: bool = True, axis: int = -1, name: str = None):
        super(BatchNormalization, self).__init__(key=key, trainable=trainable)
        self.axis = axis
        self.name = name
    
    def build(self, input_shape: tuple):
        init_fun, self.apply_fn = stax.BatchNorm(axis=self.axis)
        self._check_jit()
        self.shape, self._params = init_fun(rng=self.key, input_shape=input_shape)
        self.input_shape = self.shape
        self.built = True
    
    def call(self, inputs: DeviceArray):
        self.output = self.apply_fn(params=self._params, x=inputs)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output = self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __repr__(self):
        if self.built:
            return f"<Batch Normalization Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Batch Normalization Layer>"
