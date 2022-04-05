from typing import Callable, Union

from jax import jit, lax
from jax import numpy as jnp
from jax import random, vmap
from jax._src.lax.convolution import ConvDimensionNumbers
from jax.numpy import DeviceArray
from jax.random import PRNGKey  # type: ignore

from .core import Layer


class Conv2D(Layer):
    '''
    Convolutional Layer, (Layer Subclass)
    Params:
        - filters: Stores number of filters, accepts int
        - kernel_size: stores size of each filter, accepts int or tuple
        - strides: stores size of the strides, default (1,1), accepts int or tuple
        - padding: padding for the input, accepts "valid" or "same"
        - activation: stores the activation function, accepts activation as string or callable
        - kernel_initializer: stores the kernel initializer, default "glorot_uniform"
        - bias_initializer: stores the bias initializer, default "zeros"
        - key: stores Pseudo Random Generator Key, default PRNGKey(1)
        - input_dim_order: stores the order of the dimensions, default NHWC
            where N=Batch size, H=Height, W=Width and C=Number of channels
        - kernel_dim_order: stores the order of the dimensions, default HWIO
            where H=Height, W=Width, I=Input Size which is the number of channels of the input and O=Output size which is the number of the filters
        - output_dim_order: stores the order of the dimensions, default NHWC

    '''
    def __init__(self, filters: int, kernel_size: Union[int, tuple], 
                 strides: Union[int, tuple] = (1,1), padding: str = 'valid', activation: Union[str, Callable] = None, 
                 kernel_initializer: Union[str, Callable] ='glorot_uniform', bias_initializer: Union[str, Callable] ='normal',
                 use_bias: bool = False, key: PRNGKey = PRNGKey(100), 
                 input_dim_order: str = "NHWC", kernel_dim_order: str = "HWIO", output_dim_order: str = "NHWC", 
                 trainable: bool = True, dtype='float32', name: str = None, *args, **kwargs):
        super(Conv2D, self).__init__(key=key, trainable=trainable, dtype=dtype, name=name, *args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.built = False
        self.use_bias = use_bias
        self.dimension_numbers = (input_dim_order, kernel_dim_order, output_dim_order)
        self._validate_init()
        shape = kwargs.get('shape', False) or kwargs.pop('input_shape', False)
        if shape:
            self.build(shape)
    
    @property
    def kernel_shape(self):
        'Returns the kernel dimensions'
        return self.kernel_weights.shape

    @property
    def bias_shape(self):
        'Returns the bias dimensions'
        if self.use_bias:
            return self.bias_weights.shape
        else:
            raise Exception('Bias is turned OFF, set use_bias=True in the constructor to use it')

    def compute_output_shape(self):
        if self.built:
            return lax.conv_general_shape_tuple(self.input_shape, self.kernel_weights.shape, self.strides, self.padding, self.dimension_numbers)
        else:
            raise Exception(f"{self.name} is not built yet, use call() or build() to build it.")

    @property
    def shape(self):
        return self.compute_output_shape()

    def compute_kernel_shape(self, input_shape: tuple):
        return (*self.kernel_size, input_shape[-1], self.filters)
    
    def compute_bias_shape(self):
        return (self.filters,)

    def _validate_init(self):
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides *= 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 1:
            self.kernel_size *= 2

    def build(self, input_shape: tuple):
        'Initializes the Kernel and stores the Conv2D weights'
        if len(input_shape) == 3:
            input_shape = (None, *input_shape)
        
        k1, k2 = random.split(self.key)
        kernel_shape = self.compute_kernel_shape(input_shape)
        self.kernel_weights = self.add_weight(k1, kernel_shape, self.kernel_initializer, self.dtype, f'{self.name}_kernel', self.trainable)
        if self.use_bias:
            bias_shape = self.compute_bias_shape()
            self.bias_weights = self.add_weight(k2, bias_shape, self.bias_initializer, self.dtype, f'{self.name}_bias', self.trainable)

        self._input_shape = input_shape
        self.dn = lax.conv_dimension_numbers(input_shape, kernel_shape, self.dimension_numbers)
        self.built = True

    def convolution_op(self, params: tuple, inputs: DeviceArray):
        output = lax.conv_general_dilated(inputs, params[0], self.strides, self.padding, dimension_numbers=self.dn)
        if self.use_bias:
            output = jnp.add(output, params[1])
        return output

    def call(self, inputs: DeviceArray, **kwargs):
        self.output = self.convolution_op(self.params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray, **kwargs):
        self.output =  self.convolution_op(params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

class MaxPool2D(Layer):
    '''
    MaxPool Layer, (Layer subclass)
    Params:
        - pool_size: takes the pooling size, default (2,2), accepts int or tuple
        - strides: stores size of the strides, default (1,1), accepts int or tuple
        - padding: padding for the input, accepts "valid" or "same"
        - spec: store the layer specs
        - key: stores Pseudo Random Generator Key, default PRNGKey(1)
    '''

    def __init__(self, pool_size: Union[int, tuple] = (2,2), strides: Union[int, tuple] = (2,2), padding: str = 'valid', 
    spec: ConvDimensionNumbers = None, key: PRNGKey = PRNGKey(1), *args, **kwargs):
        super(MaxPool2D, self).__init__(key=key, trainable=False, *args, **kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.spec = spec
        self.trainable = False

        input_shape = kwargs.get('input_shape', False)
        if input_shape:
            self.build(input_shape)
        
    def _validate_init(self):
        if isinstance(self.pool_size, int):
            self.pool_size = (self.pool_size, self.pool_size)
            
        elif isinstance(self.pool_size, tuple) and len(self.pool_size) == 1:
            self.pool_size += self.pool_size

        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides += self.strides

        non_spatial_axes = 0, len(self.pool_size) + 1
        for i in sorted(non_spatial_axes):
            window_shape = self.pool_size[:i] + (1,) + self.pool_size[i:]
            strides = self.strides[:i] + (1,) + self.strides[i:]

        self.pool_size = window_shape
        self.strides = strides
    
    def compute_output_shape(self):
        padding_vals = lax.padtype_to_pads(self.input_shape, self.pool_size,
                                         self.strides, self.padding)
        ones = (1,) * len(self.pool_size)
        out_shape = lax.reduce_window_shape_tuple(self.input_shape, self.pool_size, self.strides, padding_vals, ones, ones)
        return out_shape

    @property
    def shape(self):
        return self.compute_output_shape()

    def build(self, input_shape: tuple):
        self.input_shape = input_shape
        self.built = True

    def maxpool_op(self, params: tuple, inputs: DeviceArray):
        out = lax.reduce_window(inputs, -jnp.inf, lax.add, self.pool_size, self.strides, self.padding)
        return out

    def call(self, inputs: DeviceArray, *args, **kwargs):
        self.output = self.maxpool_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: tuple, inputs: DeviceArray, *args, **kwargs):
        self.output = self.maxpool_op(params, inputs)
        return self.output

class AvgPool2D(Layer):
    pass

class Conv2DTranspose(Layer):
    pass

