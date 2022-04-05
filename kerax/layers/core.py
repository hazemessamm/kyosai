import operator as op
from functools import reduce
from typing import Any, Callable, Tuple, Union

from jax import lax, jit
from jax import numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kerax import activations, backend
from kerax.engine import Trackable
from kerax.engine.containers import NodeContainer, Weight
from kerax.initializers import Initializer, initializers
from numpy import ndarray




class InputSpec:
    def __init__(self):
        self.input_shape = None
        self.valid_ndims = None
        self._internal_input_shape = None

    def build(self, input_shape, valid_ndims):
        if valid_ndims != len(input_shape):
            raise Exception(f'number of dims in input_shape does not match the required number of dims, found {len(input_shape)} expected {valid_ndims}')
        
        self.input_shape = input_shape
        self.valid_ndims = valid_ndims
        self._internal_input_shape = (None, *input_shape)



class Layer(Trackable):
    def __init__(self, key: PRNGKey = False, trainable: bool = True, dtype: str = 'float32', name: str = None, *args, **kwargs):
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
        self._has_nested_layers = False
        self._is_nested = False
        self._requires_unpacking_on_call = False


    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Layer):
            if not self._has_nested_layers:
                self._has_nested_layers = True
                self._nested_layers = [__value]
            else:
                self._nested_layers.append(__value)
            
            __value._is_nested = True
        return super().__setattr__(__name, __value)

    def __name__(self):
        return self.name

    def check_jit(self):
        if backend.is_jit_enabled():
            self.call = jit(self.call)


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

    @property
    def params(self):
        params = [param.get_weights() for param in self._params]
        if self._has_nested_layers:
            nested_params = tuple(layer.params for layer in self._nested_layers)
            params.extend(nested_params)
        return tuple(params)

    @property
    def named_params(self):
        return {param.name: param.get_weights for param in self._params}

    def build(self, input_shape: Tuple):
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

    def add_weight(self, key: PRNGKey, shape: Tuple, initializer: Initializer, dtype: str, name: str, trainable: bool):
        weight = Weight(key, shape, initializer, dtype, name, trainable)
        self._params.append(weight)
        return weight

    def get_weights(self):
        return self._params

    def set_weights(self, new_weights: Tuple):
        for w1, w2 in zip(self._params, new_weights):
            w1.set_weights(w2)

    def update_weights(self, new_weights: Tuple):
        if self.trainable and len(new_weights) > 0:
            for w1, w2 in zip(self._params, new_weights):
                if w1.trainable:
                    w1.set_weights(w2)

    def __call__(self, inputs, *args, **kwargs):
            if isinstance(inputs, (list, tuple)):
                if self.built:
                    return self.call(inputs, *args, **kwargs)
                
                # if the inputs are a list of layers
                if all([isinstance(_input, Layer) for _input in inputs]):
                    self.build([_input.shape for _input in inputs])
                    self.connect(inputs)
                    return self

                # if the inputs are list of tensors
                elif all([isinstance(_input, (ndarray, DeviceArray)) for _input in inputs]):
                    self.build([_input.shape for _input in inputs])
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

    def call(self, inputs: DeviceArray):
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


    def __init__(self, shape: Tuple = None, dtype: str = 'float32', name: str = None):
        super().__init__(key=None, trainable=False, dtype=dtype, name=name)
        if not shape or not isinstance(shape, tuple):
            raise Exception(f'shape should have value in a tuple, found {shape} with type {type(shape)}')
        if shape:
            shape = tuple(shape)
            self._shape, self._input_shape = shape, shape
            self.built = True
        else:
            raise Exception(f'Error in {self.name}, input shape must be provided for the Input Layer')
    
    @property
    def shape(self):
        return (None, *self._shape)

    @property
    def input_shape(self):
        return self._input_shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape

    def call(self, inputs: DeviceArray):
        self.output = inputs
        return inputs

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output = inputs
        return inputs

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
    bias_initializer: Union[str, Callable] = 'normal', use_bias: bool = True, trainable: bool = True, key: PRNGKey = PRNGKey(100), dtype: str = 'float32', *args, **kwargs):
        super(Dense, self).__init__(key=key, trainable=trainable, dtype=dtype, *args, **kwargs)
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
        return (None, self.units)

    def compute_kernel_shape(self):
        return self.kernel_weights.shape

    def compute_bias_shape(self):
        return self.bias_weights.shape

    def compute_output_shape(self):
        return (None, self.units)

    def build(self, input_shape: Tuple):
        self._input_shape = (input_shape[-1],)
        k1, k2 = random.split(self.key)
        self.kernel_weights = self.add_weight(k1, (input_shape[-1], self.units), self.kernel_initializer, self.dtype, f'{self.name}_kernel', trainable=self.trainable)
        if self.use_bias:
            self.bias_weights = self.add_weight(k2, (self.units,), self.bias_initializer, self.dtype, f'{self.name}_bias', trainable=self.trainable)
        self.built = True

    def dense_op(self, params: Tuple, inputs: DeviceArray):
        out = jnp.matmul(inputs, params[0])
        if self.use_bias:
            out = jnp.add(out, params[1])
        return out

    def call(self, inputs: DeviceArray):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.dense_op(self.params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray):
        self.output = self.dense_op(params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

class Flatten(Layer):
    '''
    Flatten Layer, (Layer subclass)
    params: 
    key: Pseudo Random Generator Key, default PRNGKey(1)

    '''
    def __init__(self, *args, **kwargs):
        super(Flatten, self).__init__(key=None, trainable=False, *args, **kwargs)

    @property
    def compute_output_shape(self):
        return (self.input_shape[0], reduce(op.mul, self.input_shape[1:], 1))

    @property
    def shape(self):
        return (None, reduce(op.mul, self.input_shape[1:], 1))

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def flatten_op(self, params: Tuple, inputs: DeviceArray):
        return lax.reshape(inputs, (inputs.shape[0], *self.shape[1:]))

    def call(self, inputs: DeviceArray):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.flatten_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output =  self.flatten_op(params, inputs)
        return self.output

class Dropout(Layer):
    '''
    Dropout Layer, (Layer subclass)
    params:
    rate: probability of turning of a neuron, accepts probability values between 0 and 1
    training: stores the mode of the layer, accepts boolean
    key: Pseudo Random Generator Key, default PRNGKey(1)
    '''
    def __init__(self, rate: float, key: PRNGKey = PRNGKey(1), *args, **kwargs):
        super(Dropout, self).__init__(key=key, *args, **kwargs)
        self.rate = rate
        self.key = key
    
    @property
    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._input_shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def dropout_op(self, params: Tuple, inputs: DeviceArray):
        keep = random.bernoulli(self.key, self.rate, inputs.shape)
        return jnp.where(keep, inputs / self.rate, 0)

    def call(self, inputs: DeviceArray, *args, **kwargs):
        'Used during training to pass the parameters while getting the gradients'
        training = kwargs.get('training', False)
        if training:
            self.output = self.dropout_op(self.params, inputs)
        else:
            self.output = inputs
        return self.output
    
    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output = self.dropout_op(params, inputs)
        return self.output

class Activation(Layer):
    '''
    Activation Layer, (Layer subclass)
    params:
    identifier: accepts the activation function as a string or callable
    '''
    def __init__(self, identifier: Union[str, Callable], *args, **kwargs):
        super(Activation, self).__init__(key=None, *args, **kwargs)
        self._identifier = identifier
        self.activation = activations.get(identifier)

    @property
    def identifier(self):
        return self._identifier

    @property
    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._input_shape
    
    def activation_op(self, params: Tuple, inputs: DeviceArray):
        return self.activation(inputs)

    def build(self, input_shape: tuple):
        self._input_shape = input_shape
        self.built = True

    def call(self, inputs: DeviceArray):
        self.output = self.activation_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output =  self.activation_op(params, inputs)
        return self.output


class Concatenate(Layer):
    def __init__(self, axis: int = -1, name: str = None, *args, **kwargs):
        super(Concatenate, self).__init__(name=name, *args, **kwargs)
        self.axis = axis

    @property
    def shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def build(self, input_shape: Tuple):
        if not isinstance(input_shape, list) or len(input_shape) <= 1:
            raise Exception('Input shapes should be passed as a list with more than one value in it to the build() method')
        else:
            first_shape = input_shape[0][self.axis]
            for curr_shape in input_shape[1:]:
                if curr_shape[self.axis] != first_shape:
                    raise Exception(f'Input shapes should have the same dimension at axis {self.axis}')

        self._input_shape = input_shape[0]
        self._output_shape = (*input_shape[0][:-1], sum([i[self.axis] for i in input_shape]))
        self.built = True
    
    def concatenate_op(self, params: Tuple, inputs: DeviceArray):
        return jnp.concatenate(inputs, axis=self.axis)

    def call(self, inputs: DeviceArray):
        self.output = self.concatenate_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output = self.concatenate_op(params, inputs)
        return self.output


class Add(Layer):
    def __init__(self, name: str = None, *args, **kwargs):
        super(Add, self).__init__(name=name, *args, **kwargs)

    @property
    def shape(self):
        return self._output_shape

    @property
    def compute_output_shape(self):
        return self._output_shape

    def add_op(self, params: Tuple, inputs: DeviceArray):
        inputs = jnp.stack(inputs, axis=0)
        return jnp.sum(inputs, axis=0)

    def build(self, input_shape: Tuple):
        if not isinstance(input_shape, list) or len(input_shape) <= 1:
            raise Exception('Input shapes should be passed as a list with more than one value in it to the build() method')
        else:
            first_shape = input_shape[0]
            for curr_shape in input_shape[1:]:
                if curr_shape != first_shape:
                    raise Exception('Input shapes should have the same last dimension')

        self._input_shape, self._output_shape = input_shape[0], input_shape[0]
        self.built = True

    def call(self, inputs: DeviceArray):
        self.output = self.add_op(self._params, inputs)
        return self.output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        out = inputs[0]
        for _input in inputs[1:]:
            out = self.add_op(params, (out, _input))
        self.output = out
        return self.output

class BatchNormalization(Layer):
    def __init__(self, key: PRNGKey = PRNGKey(100), trainable: bool = True, axis: int = -1, name: str = None, *args, **kwargs):
        super(BatchNormalization, self).__init__(key=key, trainable=trainable, *args, **kwargs)
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

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output = self.apply_fn(params=params, inputs=inputs)
        return self.output
