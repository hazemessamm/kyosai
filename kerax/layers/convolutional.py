from jax import lax, nn
from jax.experimental import stax
from jax import numpy as jnp
from jax import jit
from jax.random import PRNGKey
import core
import numpy as np

class Conv2D(core.Layer):
    def __init__(self, filters, kernel_size, 
                 strides=(1,1), padding='valid', activation=None, input_shape=None, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 key=PRNGKey(1), input_dim_order="NHWC", kernel_dim_order="HWIO", output_dim_order="NHWC"):
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
        self.is_built = False
        self.dimension_numbers = (input_dim_order, kernel_dim_order, output_dim_order)
        self.key = key
        self.init_fn, self.apply_fn = stax.GeneralConv(dimension_numbers=self.dimension_numbers, 
        filter_shape=self.kernel_size, padding=padding, out_chan=filters, W_init=self.kernel_initializer, b_init=self.bias_initializer)
        self.apply_fn = jit(self.apply_fn)


    def get_kernel_shape(self):
        return self.kernel_shape
    
    def get_bias_shape(self):
        return self.bias_shape
    
    def check_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            
    def check_strides(self, strides):
        if isinstance(strides, int):
            self.strides = (strides, strides)

    def build(self, input_shape):
        if len(input_shape) == 3:
            input_shape = (0,) + input_shape
        elif len(input_shape) < 3:
            raise Exception(f'Expected input shape to be at least 3 dimensions found {len(input_shape)} dimensions')

        output_shape, params = self.init_fn(input_shape=input_shape, rng=self.key)
        self.params = params
        self.input_shape = input_shape
        self.kernel_shape = params[0].shape
        self.bias_shape = params[1].shape
        self.is_built = True
        self.shape = output_shape

    def call_with_external_weights(self, inputs, params):
        'Used during training to pass the parameters while getting the gradients'
        out = self.apply_fn(inputs=inputs, params=params)
        return self.activation(out) if self.activation is not None else out

    
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
        if self.is_built:
            return f"Convolutional Layer with input shape {self.input_shape} and output shape {self.shape}"
        else:
            return "Convolutional Layer"
