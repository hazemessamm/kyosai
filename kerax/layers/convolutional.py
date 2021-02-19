import set_path
from jax import lax, nn
from jax.experimental import stax
from jax import numpy as jnp
from jax import jit
from jax.random import PRNGKey
import layers.core as core
import numpy as np



class Conv2D(core.Layer):

    '''

    Convolutional Layer, (Layer Subclass)
    Params:
    
    filters: Stores number of filters, accepts int
    
    kernel_size: stores size of each filter, accepts int or tuple
    
    strides: stores size of the strides, default (1,1), accepts int or tuple
    
    padding: padding for the input, accepts "valid" or "same"
    
    activation: stores the activation function, accepts activation as string or callable
    
    kernel_initializer: stores the kernel initializer, default "glorot_uniform"
    
    bias_initializer: stores the bias initializer, default "zeros"
    
    key: stores Pseudo Random Generator Key, default PRNGKey(1)
    
    input_dim_order: stores the order of the dimensions, default NHWC
    where N=Batch size, H=Height, W=Width and C=Number of channels

    kernel_dim_order: stores the order of the dimensions, default HWIO
    where H=Height, W=Width, I=Input Size which is the number of channels of the input and
    O=Output size which is the number of the filters

    output_dim_order: stores the order of the dimensions, default NHWC

    '''
    def __init__(self, filters, kernel_size, 
                 strides=(1,1), padding='valid', activation=None, 
                 kernel_initializer='glorot_uniform', bias_initializer='normal', 
                 key=PRNGKey(10), input_dim_order="NHWC", kernel_dim_order="HWIO", output_dim_order="NHWC"):
        super(Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = self.get_activation(activation)
        self.check_kernel_size(kernel_size)
        self.check_strides(strides)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.built = False
        self.dimension_numbers = (input_dim_order, kernel_dim_order, output_dim_order)
        self.key = key
        self.init_fn, self.apply_fn = stax.GeneralConv(dimension_numbers=self.dimension_numbers, 
        filter_shape=self.kernel_size, padding=padding, out_chan=filters, W_init=self.kernel_initializer, b_init=self.bias_initializer)
        self.apply_fn = jit(self.apply_fn)


    def get_kernel_shape(self):
        'Returns the kernel dimensions'
        return self.kernel_shape
    
    def get_bias_shape(self):
        'Returns the bias dimensions'
        return self.bias_shape
    
    def check_kernel_size(self, kernel_size):
        'Checks for the kernel size because it can be int or tuple of size one but JAX accepts only tuple'
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 1:
            self.kernel_size += kernel_size
            
    def check_strides(self, strides):
        'Checks for the strides because it can be int or tuple of size one but JAX accepts only tuple'
        if isinstance(strides, int):
            self.strides = (strides, strides)
        elif isinstance(strides, tuple) and len(strides) == 1:
            self.strides += strides

    def build(self, input_shape):
        'Initializes the Kernel and stores the Conv2D weights'
        if len(input_shape) == 3:
            input_shape = (0,) + input_shape
        elif len(input_shape) < 3:
            raise Exception(f'Expected input shape to be at least 3 dimensions found {len(input_shape)} dimensions')

        #initializes the conv layer
        self.shape, self.params = self.init_fn(input_shape=input_shape, rng=self.key)
        self.input_shape = input_shape
        self.kernel_shape = self.params[0].shape
        self.bias_shape = self.params[1].shape
        self.built = True

    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=params)
        return self.activation(out) if self.activation is not None else out

    
    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            #here it takes the previous layer to build the current layer
            if isinstance(inputs, core.Layer) or isinstance(inputs, core.Input):
                self.build(inputs.shape)
                #General function used to connect with the previous layer
                self.connect(inputs)
                #returns the current layer
                return self
            else:
                #if the inputs are tensors and the function it will be built and continue apply conv to it
                if not self.built:
                    self.build(inputs.shape)
                
                if self.input_shape[1:] != inputs.shape[1:]:
                    raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
                else:
                    out = self.apply_fn(inputs=inputs, params=self.params)
                    return self.activation(out) if self.activation is not None else out

    def __repr__(self):
        if self.built:
            return f"<Convolutional Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Convolutional Layer>"

class MaxPool2D(core.Layer):


    '''

    MaxPool Layer, (Layer subclass)
    params:
    pool_size: takes the pooling size, default (2,2), accepts int or tuple
    
    strides: stores size of the strides, default (1,1), accepts int or tuple
    
    padding: padding for the input, accepts "valid" or "same"

    spec: store the layer specs
    
    key: stores Pseudo Random Generator Key, default PRNGKey(1)



    '''

    def __init__(self, pool_size=(2,2), strides=None, padding='valid', spec=None, key=PRNGKey(1)):
        self.pool_size = pool_size
        self.strides = strides
        self.check_pool_size(pool_size)
        self.check_strides(strides)
        self.padding = padding
        self.spec = spec
        self.key = key
        #initializing maxpool
        self.init_fn, self.apply_fn = stax.MaxPool(window_shape=pool_size, padding=padding, strides=strides, spec=spec)



    def check_pool_size(self, pool_size):
        'Checks for the pool size because it can be int or tuple of size one but JAX accepts only tuple'
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        elif isinstance(pool_size, tuple) and len(pool_size) == 1:
            self.pool_size += pool_size

    def check_strides(self, strides):
        'Checks for the strides because it can be int or tuple of size one but JAX accepts only tuple'
        if isinstance(strides, int):
            self.strides = (strides, strides)
        elif isinstance(strides, tuple) and len(strides) == 1:
            self.strides += strides
    
    def build(self, input_shape):
        #returns output shape, and the params
        self.shape, self.params = self.init_fn(input_shape=input_shape, rng=self.key)
        self.input_shape = input_shape
        self.built=True

    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=self.params)
        return out
    
    def __call__(self, inputs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        else:
            if isinstance(inputs, core.Layer) or isinstance(inputs, core.Input):
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
            return f"<MaxPool Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<MaxPool Layer>"