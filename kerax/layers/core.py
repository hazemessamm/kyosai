import initializers
import activations
from jax import numpy as jnp #type: ignore
from jax import jit #type: ignore
from jax.experimental import stax #type: ignore
from jax.random import PRNGKey, split #type: ignore
from . import construction_layers
from jaxlib.xla_extension import DeviceArray #type: ignore


class Input:
    '''
    Input Layer that stores the training data and returns batches from it 
    params:
    shape: takes a tuple (0, H, W, C)
    '''


    def __init__(self, shape=None):
        if not shape or not isinstance(shape, tuple):
            raise Exception(f'shape should have value in a tuple, found {shape}')
        self.shape = shape
        #stores the index state
        self.training_index = 0
    

    def get_shape(self):
        return self.shape

    def set_batch_size(self, batch_size):
        'Stores the batch size'
        if batch_size > 0:
            self.batch_size = batch_size
        else:
            raise Exception('Batch size should be bigger than zero')

    def store_data(self, x, y, **kwargs):
        'stores the data also accepts validation_data '
        if not hasattr(self, 'batch_size'):
            raise Exception('batch size should have a value, use set_batch_size() and pass int value')
        
        if len(x) < self.batch_size:
            raise Exception('batch size should be smaller than the training data')

        if len(x) != len(y):
            raise Exception(f"Training data should be equal training labels, {len(x)} != {len(y)}")
        
        self.training_data = x
        self.training_labels = y
        self.data_length = len(x)
        self.num_batches = self.data_length // self.batch_size
        self.built=True
        if kwargs.get('validation_data', None) is not None:
            self.validation_data, self.validation_labels = kwargs.get('validation_data')
            if len(self.validation_data) != len(self.validation_labels):
                raise Exception(f'Validation data should be equal validation labels, {len(x)} != {len(y)}')
            self.validation_length = len(self.validation_data)
            self.validation_index = 0
            

    def check_index_range(self, index_range, required_length):
        return index_range > required_length
    
    def __call__(self, x):
        return x

    def get_training_batch(self):
        if hasattr(self, 'built'):
            current_batch_index = self.training_index*self.batch_size
            status = self.check_index_range(current_batch_index+self.batch_size, self.data_length)
            if status:
                self.training_index = 0
                current_batch_index = 0
            current_batch_x = self.training_data[current_batch_index: current_batch_index+self.batch_size]
            current_batch_y = self.training_labels[current_batch_index: current_batch_index+self.batch_size]
            self.training_index += 1
            return current_batch_x, current_batch_y

    def get_validation_batch(self):
        if not hasattr(self, 'validation_index'):
            raise Exception('Validation data is not initialized')
        current_batch_index = self.validation_index*self.batch_size
        status = self.check_index_range(current_batch_index+self.batch_size, self.validation_length)
        if status:
            self.validation_index = 0
            current_batch_index = 0
        current_batch_x = self.validation_data[current_batch_index: current_batch_index+self.batch_size]
        current_batch_y = self.validation_labels[current_batch_index: current_batch_index+self.batch_size]
        self.validation_index += 1
        return current_batch_x, current_batch_y

    def __repr__(self):
        return "<Input Layer>"

class Layer:
    def __init__(self, key=False, trainable=True, name=None):
        #stores the layer params
        self.params = ()
        #stores the previous layer
        self.prev = None
        self.built = False
        if not isinstance(key, DeviceArray):
            key = PRNGKey(key)
        self.key, self.subkey = split(key)
        self.trainable = trainable
        self.name = self.__class__.__name__ if name is None else name
        self.output = None
        self.init_fn, self.apply_fn = None, None

    def get_initializer(self, identifier):
        'Returns the specified initializer'
        return initializers.get(identifier)

    def get_activation(self, identifier):
        'Returns the specified activation'
        return activations.get(identifier)

    def connect(self, layer):
        'Connects the current layer with the previous layer'
        self.prev = layer
    
    def get_weights(self):
        'returns weights'
        return self.params

    def set_weights(self, new_weights):
        if not isinstance(new_weights, tuple):
            raise Exception(f"Weights should be inside a tuple example: (W, b), found {type(new_weights)}")
        #looping over the current weights and the new weights
        #checks the shape of each dimension
        #finally stores the new weights
        for current_p, new_p in zip(self.params, new_weights):
            if current_p.shape != new_p.shape:
                raise Exception(f"New weights are not compatible with the current weight shapes, {current_p.shape} != {new_p.shape}")
        
        self.params = new_weights

    def update_weights(self, new_weights):
        if self.trainable:
            self.set_weights(new_weights)

    def call(params, inputs):
        raise NotImplementedError('This method should be implemented in Layer subclasses')

    def get_output(self):
        if self.output is None:
            raise Exception('This layer does not have an output yet, call method should be called')
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output


class Dense(Layer):
    '''

    Dense Layer, (Layer subclass)

    params:
    units: stores number of columns (neurons)
    activation: stores the activation function, default None
    kernel_initializer: stores the kernel initializer, default "glorot_normal"
    bias_initializer: stores the bias initializer, default "normal"

    '''
    def __init__(self, units, activation=None, kernel_initializer='glorot_normal', bias_initializer='normal', 
    input_shape=None, trainable=True, key=PRNGKey(100)):
        super(Dense, self).__init__(key=key, trainable=trainable)
        self.units = units
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.init_fn, self.apply_fn = stax.Dense(units, W_init=self.kernel_initializer, b_init=self.bias_initializer)
        self.apply_fn = jit(self.apply_fn)
    
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

    def call_with_external_weights(self, params, inputs):
        self.output = super().call_with_external_weights(params, inputs)
        return self.activation(self.output) if self.activation is not None else self.output

    def call(self, inputs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.apply_fn(inputs=inputs, params=self.params)
        return self.activation(self.output) if self.activation is not None else self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
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
        #initializes flatten layer
        #returns initialization function and apply function
        self.init_fn, self.apply_fn = stax.Flatten
        self.apply_fn = jit(self.apply_fn)

    def build(self, input_shape):
        self.shape, self.params = self.init_fn(input_shape=input_shape, rng=self.key)
        self.input_shape = input_shape
        self.built = True

    def call(self, inputs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.apply_fn(inputs=inputs, params=self.params)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs, self.params)

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
        self.init_fn, self.apply_fn = stax.Dropout(rate=rate, mode='train')
        self.apply_fn = jit(self.apply_fn)
    
    def call(self, inputs, training=False):
        'Used during training to pass the parameters while getting the gradients'
        self.output = inputs
        if training:
            self.output = self.apply_fn(inputs=inputs, params=self.params)
        return self.output
    
    def build(self, input_shape):
        self.shape, self.params = self.init_fn(rng=self.key, input_shape=input_shape)
        self.input_shape = input_shape
        self.kernel_shape = self.params[0].shape
        self.bias_shape = self.params[-1].shape
        self.built = True

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
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
        self.identifier = identifier
        self.init_fn, self.apply_fn = construction_layers.Activation(identifier)
    
    def build(self, input_shape):
        self.shape, self.params = self.init_fn(input_shape)
        self.input_shape = self.shape
        self.built=True

    def call(self, inputs):
        self.output = self.apply_fn(self.params, inputs)
        return self.output

    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
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
            return f"<{self.identifier} Activation Layer with input shape {self.input_shape} and output shape {self.shape}>"


class Concatenate(Layer):
    def __init__(self, layers, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis
        self.layers = layers

    def call(self, inputs):
        self.output = self.layers[0].get_output()
        for layer in self.layers[1:]:
            self.output = jnp.concatenate((self.output, layer.get_output()), axis=self.axis)
        return self.output
    
    def build(self, input_shape):
        self.params = ()
        self.shape = input_shape
        self.built = True
        self.input_shape = input_shape
    
    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
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
    def __init__(self, layers, name=None):
        super(Add, self).__init__(name=name)
        self.layers = layers
        self.init_fn, self.apply_fn = construction_layers.Add()
    
    def build(self, input_shape):
        self.shape, self.params = self.init_fn(input_shape, self.key)
        self.input_shape = self.shape
        self.built = True

    def call_with_external_weights(self, params, inputs):
        out = self.layers[0].get_output()
        for i in range(1, len(self.layers)):
            if self.layers[i].output is None:
                prev_layer_output = self.layers[i].prev.get_output()
                self.layers[i](prev_layer_output)
            out = self.apply_fn(params, out, self.layers[i].get_output())
        self.output = out
        return self.output


    def call(self, inputs):
        self.output = self.layers[0].get_output()
        for layer in self.layers[1:]:
            self.output = self.apply_fn(self.params, self.output, layer.get_output())
        return self.output
        

    def __call__(self, inputs, params=None):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)

class BatchNormalization(Layer):
    def __init__(self, key=PRNGKey(100), trainable=True, axis=-1, name=None):
        super(BatchNormalization, self).__init__(key=key, trainable=trainable)
        self.axis = axis
        self.name = name
        self.init_fun, self.apply_fn = stax.BatchNorm(axis=axis)
    
    def build(self, input_shape):
        self.shape, self.params = self.init_fun(rng=self.key, input_shape=input_shape)
        self.input_shape = self.shape
        self.built=True
    
    def call(self, inputs):
        self.output = self.apply_fn(params=self.params, x=inputs)
        return self.output
    
    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if isinstance(inputs, (Layer, Input)):
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
