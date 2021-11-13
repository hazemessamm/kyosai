from kerax.engine import Trackable
from kerax.initializers import initializers
from kerax import activations
from jax import numpy as jnp #type: ignore
from jax import jit #type: ignore
from jax.experimental import stax #type: ignore
from jax.random import PRNGKey, split #type: ignore
from kerax.layers import construction_layers
from jaxlib.xla_extension import DeviceArray #type: ignore
import kerax.jit as jit_execution


class Layer:
    def __init__(self, key=False, trainable=True, name=None):
        #stores the layer params
        self._params = ()
        #Stores the previous layer
        self.prev = []
        self.next = []
        self.built = False
        if key is not None:
            if not isinstance(key, DeviceArray):
                key = PRNGKey(key)
            self.key, self.subkey = split(key)
        self.trainable = trainable
        self.__name = Trackable.get_uid(self.__class__.__name__ if name is None else name)
        self.output = None
        self.apply_fn = None 
        self.index = Trackable.depth
        Trackable.track_layer(self.__name, self)

    def _check_jit(self):
        if jit_execution.is_jit_enabled():
            self.apply_fn = jit(self.apply_fn)

    def in_construction_mode(self, layer):
        if isinstance(layer, Layer):
            return True

    @property
    def name(self):
        return self.__name

    def get_initializer(self, identifier):
        'Returns the specified initializer'
        return initializers.get(identifier)

    def get_activation(self, identifier):
        'Returns the specified activation'
        return activations.get(identifier)

    def connect(self, layer):
        'Connects the current layer with the previous layer'
        if isinstance(layer, (list, tuple)):
            self.prev += layer
        else:
            self.prev.append(layer)

        if isinstance(layer, (list, tuple)):
            for l in layer:
                l.next.append(self)
        else:
            layer.next.append(self)
        
    
    @property
    def weights(self):
        'returns weights'
        return self._params
    
    @property
    def params(self):
        return self._params

    def get_weights(self):
        return self._params

    def set_weights(self, new_weights):
        if not isinstance(new_weights, tuple):
            raise Exception(f"Weights should be inside a tuple example: (W, b), found {type(new_weights)}")
        #looping over the current weights and the new weights
        #checks the shape of each dimension
        #finally stores the new weights
        for current_p, new_p in zip(self._params, new_weights):
            if current_p.shape != new_p.shape:
                raise Exception(f"New weights are not compatible with the current weight shapes, {current_p.shape} != {new_p.shape}")        
        self._params = new_weights

    def update_weights(self, new_weights):
        if self.trainable:
            self.set_weights(new_weights)

    def call(params, inputs):
        raise NotImplementedError('This method should be implemented in Layer subclasses')

    def get_output(self):
        if self.output is None:
            raise Exception('This layer does not have an output yet, call method should be called')
        return self.output


class Input(Layer):
    '''
    Input Layer that stores the training data and returns batches from it 
    params:
    shape: takes a tuple (0, H, W, C)
    '''


    def __init__(self, shape=None, name=None):
        super().__init__(key=None, trainable=False, name=name)
        if not shape or not isinstance(shape, tuple):
            raise Exception(f'shape should have value in a tuple, found {shape} with type {type(shape)}')
        self._shape = shape
    
    @property
    def shape(self):
        return self._shape

    def __call__(self, inputs):
        if self.in_construction_mode(inputs):
            self.connect(inputs)
        else:
            self.output = inputs
            return self.output
    

    def __repr__(self):
        return "<Input Layer>"


class Dense(Layer):
    '''

    Dense Layer, (Layer subclass)

    params:
    units: stores number of columns (neurons)
    activation: stores the activation function, default None
    kernel_initializer: stores the kernel initializer, default "glorot_normal"
    bias_initializer: stores the bias initializer, default "normal"

    '''
    def __init__(self, units, activation=None, kernel_initializer='glorot_normal', bias_initializer='normal', trainable=True, key=PRNGKey(100), **kwargs):
        super(Dense, self).__init__(key=key, trainable=trainable)
        self.units = units
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)

        input_shape = kwargs.pop('input_shape', False)
        if input_shape:
            self.build(input_shape=input_shape)
    
    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def bias_shape(self):
        return self._bias_shape

    @property
    def output_shape(self):
        return self._shape
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._shape

    def build(self, input_shape):
        init_fn, self.apply_fn = stax.Dense(self.units, W_init=self.kernel_initializer, b_init=self.bias_initializer)
        self._check_jit()
        self._shape, self._params = init_fn(rng=self.key, input_shape=input_shape)
        self._input_shape = input_shape
        self._kernel_shape = self._params[0].shape
        self._bias_shape = self._params[0].shape
        self.built = True

    def call_with_external_weights(self, params, inputs):
        self.output = self.apply_fn(params=params, inputs=inputs)
        return self.activation(self.output) if self.activation is not None else self.output

    def call(self, inputs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.apply_fn(inputs=inputs, params=self._params)
        return self.activation(self.output) if self.activation is not None else self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)

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
    def __init__(self, key=PRNGKey(1)):
        super(Flatten, self).__init__(key=key)

    @property
    def output_shape(self):
        return self._shape
    
    @property
    def input_shape(self):
        return self._input_shape

    
    @property
    def shape(self):
        return self._shape


    def build(self, input_shape):
        #initializes flatten layer
        #returns initialization function and apply function
        init_fn, self.apply_fn = stax.Flatten
        #self.apply_fn = jit(self.apply_fn)
        self._check_jit()
        self._shape, self._params = init_fn(input_shape=input_shape, rng=self.key)
        self._input_shape = input_shape
        self.built = True

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output

    def call(self, inputs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.apply_fn(inputs=inputs, params=self._params)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)

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
    def __init__(self, rate, key=PRNGKey(1)):
        super(Dropout, self).__init__(key=key)
        self.rate = rate
        self.key = key
    
    @property
    def output_shape(self):
        return self._shape
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def kernel_shape(self):
        return self._kernel_shape
    
    @property
    def bias_shape(self):
        return self._bias_shape


    def build(self, input_shape):
        self.init_fn, self.apply_fn = stax.Dropout(rate=self.rate, mode='train')
        self._check_jit()
        self._shape, self._params = self.init_fn(rng=self.key, input_shape=input_shape)
        self._input_shape = input_shape
        self._kernel_shape = self._params[0].shape
        self._bias_shape = self._params[-1].shape
        self.built = True

    def call(self, inputs, **kwargs):
        'Used during training to pass the parameters while getting the gradients'
        training = kwargs.get('training', False)
        if training:
            self.output = self.apply_fn(inputs=inputs, params=self._params)
        else:
            self.output = inputs
        return self.output
    
    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)


class Activation(Layer):
    '''
    Activation Layer, (Layer subclass)
    params:
    identifier: accepts the activation function as a string or callable
    '''
    def __init__(self, identifier):
        super(Activation, self).__init__(key=0)
        self._identifier = identifier
    

    @property
    def activation(self):
        return self._identifier

    def build(self, input_shape):
        init_fn, self.apply_fn = construction_layers.Activation(self._identifier)
        self.shape, self._params = init_fn(input_shape)
        self.input_shape = self.shape
        self.built=True

    def call(self, inputs):
        self.output = self.apply_fn(self._params, inputs)
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __call__(self, inputs, **kwargs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs, **kwargs)

    def __repr__(self):
        if self.built:
            return f"<{self._identifier} Activation Layer with input shape {self.input_shape} and output shape {self.shape}>"


class Concatenate(Layer):
    def __init__(self, layers, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis
        self.layers = layers

    def build(self, input_shape):
        self.params = ()
        self.shape = input_shape
        self.built = True
        self.input_shape = input_shape
    
    def call(self, inputs):
        self.output = self.layers[0].get_output()
        for layer in self.layers[1:]:
            self.output = jnp.concatenate((self.output, layer.get_output()), axis=self.axis)
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)


#TODO: Implmenet Add layer and BatchNorm layer
class Add(Layer):
    """
    Still working on it
    """
    def __init__(self, name=None):
        super(Add, self).__init__(name=name)

    @property
    def shape(self):
        return self._shape
    
    @property
    def input_shape(self):
        return self._input_shape


    def build(self, input_shape):
        init_fn, self.apply_fn = construction_layers.Add()
        self._shape, self._params = init_fn(input_shape, self.key)
        self._input_shape = self.shape
        self.built = True

    def call_with_external_weights(self, params, inputs):
        out = inputs[0]
        for _input in inputs[1:]:
            out = self.apply_fn(params, out, _input)
        self.output = out
        return self.output

    def call(self, inputs):
        # self.output = self.prev[0].get_output()
        # for layer in self.prev[1:]:
        #     self.output = self.apply_fn(self._params, self.output, layer.get_output())
        # return self.output
        output = inputs[0]
        for _input in inputs[1:]:
            output = self.apply_fn(self._params, output, _input)
        
        self.output = output
        return self.output

    def __call__(self, inputs, params=None):
        if isinstance(inputs, list) and all(
            isinstance(cur_input, Layer) for cur_input in inputs
        ):
            self.build(inputs[0].shape)
            self.connect(inputs)
            return self
        elif all(hasattr(cur_input, 'shape') for cur_input in inputs):
            self.call(inputs)
        else:
            raise Exception("Inputs to the Add layers should be inside a list with at least length = 2")
    
    def __repr__(self) -> str:
        if self.built:
            return f"<Add Layer with input shape {self.input_shape} and output shape {self.input_shape}>"
        else:
            return "<Add layer>"

class BatchNormalization(Layer):
    def __init__(self, key=PRNGKey(100), trainable=True, axis=-1, name=None):
        super(BatchNormalization, self).__init__(key=key, trainable=trainable)
        self.axis = axis
        self.name = name
    
    def build(self, input_shape):
        init_fun, self.apply_fn = stax.BatchNorm(axis=self.axis)
        self._check_jit()
        self.shape, self._params = init_fun(rng=self.key, input_shape=input_shape)
        self.input_shape = self.shape
        self.built = True
    
    def call(self, inputs):
        self.output = self.apply_fn(params=self._params, x=inputs)
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output = self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)
    
    def __repr__(self):
        if self.built:
            return f"<Batch Normalization Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Batch Normalization Layer>"
